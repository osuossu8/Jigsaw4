# ====================================================
# Library
# ====================================================
import os
import gc
import re
import sys
sys.path.append("/root/workspace/Jigsaw4")
import json
import time
import math
import string
import pickle
import random
import joblib
import itertools
import warnings
warnings.filterwarnings("ignore")

import scipy as sp
import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
from tqdm.auto import tqdm
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold

import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch.optim import Adam, SGD, AdamW
from torch.utils.data import DataLoader, Dataset

import transformers
from transformers import AutoTokenizer, AutoModel, AutoConfig
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup


class CFG:
    ######################
    # Globals #
    ######################
    EXP_ID = '009'
    seed = 71
    epochs = 1 # 3
    folds = [0, 1, 2, 3, 4]
    N_FOLDS = 5
    LR = 1e-4 # 2e-5
    ETA_MIN = 1e-5
    max_len = 128 # 256
    train_bs = 64 # 16 * 2
    valid_bs = 128 # 32 * 2
    log_interval = 3000
    model_name = 'roberta-base'
    EVALUATION = 'RMSE'
    EARLY_STOPPING = True
    DEBUG = False # True
    margin = 0.5
    tokenizer = AutoTokenizer.from_pretrained(model_name)
 

# ====================================================
# Directory settings
# ====================================================
import os

OUTPUT_DIR = f'output/{CFG.EXP_ID}/'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
   
    
def set_seed(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
set_seed(CFG.seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def calc_loss(y_true, y_pred):
    if CFG.EVALUATION == 'RMSE':
        return  np.sqrt(metrics.mean_squared_error(y_true, y_pred))
    elif CFG.EVALUATION == 'AUC':
        return metrics.roc_auc_score(np.array(y_true), np.array(y_pred))
    else:
        raise NotImplementedError()


def get_score(more_pred, less_pred):
    score = sum(np.array(less_pred) < np.array(more_pred)) / len(more_pred)
    return score


# ====================================================
# Training helper functions
# ====================================================
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class MetricMeter(object):
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.y_true = []
        self.y_pred = []
    
    def update(self, y_true, y_pred):
        self.y_true.extend(y_true.cpu().detach().numpy().tolist())
        self.y_pred.extend(y_pred.cpu().detach().numpy().tolist())

    @property
    def avg(self):
        self.score = calc_loss(self.y_true, self.y_pred)
       
        return {
            "score" : self.score,
        }


# ====================================================
# Data Loading
# ====================================================
# train = pd.read_csv('input/validation_data.csv')
train = pd.read_csv('input/JigsawUnintendedBias/jigsaw_unintended_bias_toxic_score.csv')
if CFG.DEBUG:
    train = train.sample(n=100, random_state=CFG.seed).reset_index(drop=True)
test = pd.read_csv('input/comments_to_score.csv')
submission = pd.read_csv('input/sample_submission.csv')
print(train.shape)
print(test.shape, submission.shape)


# ====================================================
# CV split
# ====================================================

train_over_0 = train[train['y']>0].reset_index(drop=True)
len_train_over_0 = len(train_over_0)
print(len_train_over_0)
train_0 = train[train['y']==0].sample(n=len_train_over_0, random_state=CFG.seed).reset_index(drop=True)

train = pd.concat([train_over_0, train_0], 0).reset_index(drop=True)

Fold = KFold(n_splits=CFG.N_FOLDS, random_state=42, shuffle=True)
for n, (trn_index, val_index) in enumerate(Fold.split(train, train['y'])):
    train.loc[val_index, 'kfold'] = int(n)
train['kfold'] = train['kfold'].astype(int)


train.to_csv('input/jigsaw_unintended_bias_toxic_score_folds.csv', index=False)


class Jigsaw4Dataset:
    def __init__(self, df, cfg):
        self.tokenizer = cfg.tokenizer
        self.max_len = cfg.max_len
        self.text = df['text'].fillna("none").values
        if 'y' in df.columns:
            self.target = df['y'].values 
        else:
            self.target = None

    def __len__(self):
        return len(self.text)

    def __getitem__(self, item):

        inputs = self.tokenizer(
            str(self.text[item]), 
            max_length=self.max_len, 
            padding="max_length", 
            truncation=True
        )
        
        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]
        
        if self.target is None:
            targets = torch.tensor(1, dtype=torch.float)
        else:
            targets = self.target[item]

        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.long),
            "targets" : torch.tensor(targets, dtype=torch.float32),
        }


class RoBERTaBase(nn.Module):
    def __init__(self, model_path):
        super(RoBERTaBase, self).__init__()
        self.in_features = 768
        self.roberta = AutoModel.from_pretrained(model_path)
        self.dropout = nn.Dropout(0.2)
        self.l0 = nn.Linear(self.in_features, 1)

    def forward(self, ids, mask):
        roberta_outputs = self.roberta(
            ids,
            attention_mask=mask
        )
        x = roberta_outputs[1]
        logits = torch.sigmoid(self.l0(self.dropout(x)))
        return logits.squeeze(-1)


class RMSELoss(torch.nn.Module):
    def __init__(self):
        super(RMSELoss,self).__init__()

    def forward(self,x,y):
        criterion = nn.MSELoss()
        loss = torch.sqrt(criterion(x, y))
        return loss


def loss_fn(logits, targets):
    loss_fct = RMSELoss() # torch.nn.BCEWithLogitsLoss()
    loss = loss_fct(logits, targets)
    return loss


def train_fn(epoch, model, train_data_loader, valid_data_loader, device, optimizer, scheduler, best_score):
    model.train()
    losses = AverageMeter()
    scores = MetricMeter()
    tk0 = tqdm(train_data_loader, total=len(train_data_loader))
    
    for batch_idx, data in enumerate(tk0):
        optimizer.zero_grad()
        inputs = data['input_ids'].to(device)
        masks = data['attention_mask'].to(device)
        targets = data['targets'].to(device)
        logits = model(inputs, masks)
        loss = loss_fn(logits, targets)
        loss.backward()
        optimizer.step()
        scheduler.step()
        losses.update(loss.item(), inputs.size(0))
        scores.update(targets, logits)
        tk0.set_postfix(loss=losses.avg)

        if (batch_idx > 0) and (batch_idx % CFG.log_interval == 0):
            valid_avg, valid_loss = valid_fn(model, valid_data_loader, device)

            logger.info(f"Epoch {epoch+1}, Step {batch_idx} - valid_score:{valid_avg['score']:0.5f}")

            if valid_avg['score'] < best_score:
                logger.info(f">>>>>>>> Model Improved From {best_score} ----> {valid_avg['score']}")
                torch.save(model.state_dict(), OUTPUT_DIR+f'fold-{fold}.bin')
                best_score = valid_avg['score']

            # RuntimeError: cudnn RNN backward can only be called in training mode (_cudnn_rnn_backward_input at /pytorch/aten/src/ATen/native/cudnn/RNN.cpp:877)
            # https://discuss.pytorch.org/t/pytorch-cudnn-rnn-backward-can-only-be-called-in-training-mode/80080/2
            # edge case in my code when doing eval on training step
            model.train() 

    return scores.avg, losses.avg, valid_avg, valid_loss, best_score


def valid_fn(model, data_loader, device):
    model.eval()
    losses = AverageMeter()
    scores = MetricMeter()
    tk0 = tqdm(data_loader, total=len(data_loader))

    with torch.no_grad():
        for data in tk0:
            inputs = data['input_ids'].to(device)
            masks = data['attention_mask'].to(device)
            targets = data['targets'].to(device)
            logits = model(inputs, masks)
            loss = loss_fn(logits, targets)
            losses.update(loss.item(), inputs.size(0))
            scores.update(targets, logits)
            tk0.set_postfix(loss=losses.avg)
    return scores.avg, losses.avg


def calc_cv(model_paths):
    models = []
    for p in model_paths:
        model = RoBERTaBase(CFG.model_name)
        model.to("cuda")
        model.load_state_dict(torch.load(p))
        model.eval()
        models.append(model)
     
    df = pd.read_csv("input/train_folds.csv")

    df_more = df[['more_toxic']].copy().reset_index(drop=True)
    df_less = df[['less_toxic']].copy().reset_index(drop=True)

    df_more.columns = ['text']
    df_less.columns = ['text']

    y_more = []
    y_less = []
    idx = []
    for fold, model in enumerate(models):
    
        dataset_more = Jigsaw4Dataset(df=df_more, cfg=CFG)
        data_loader_more = torch.utils.data.DataLoader(
            dataset_more, batch_size=CFG.valid_bs, num_workers=0, pin_memory=True, shuffle=False
        )

        dataset_less = Jigsaw4Dataset(df=df_less, cfg=CFG)
        data_loader_less = torch.utils.data.DataLoader(
            dataset_less, batch_size=CFG.valid_bs, num_workers=0, pin_memory=True, shuffle=False
        )

        more_output = []
        less_output = []
        for b_idx, (data1, data2) in tqdm(enumerate(zip(data_loader_more, data_loader_less))):
            with torch.no_grad():
                more_inputs = data1['input_ids'].to(device)
                more_masks = data1['attention_mask'].to(device)
                less_inputs = data2['input_ids'].to(device)
                less_masks = data2['attention_mask'].to(device)

                less_toxic_logits = model(less_inputs, less_masks)
                more_toxic_logits = model(more_inputs, more_masks)
                
                more_toxic_logits = more_toxic_logits.detach().cpu().numpy().tolist()
                less_toxic_logits = less_toxic_logits.detach().cpu().numpy().tolist()
                more_output.extend(more_toxic_logits)
                less_output.extend(less_toxic_logits)
        logger.info(get_score(np.array(more_toxic_logits), np.array(less_toxic_logits)))
        y_more.append(np.array(more_output))
        y_less.append(np.array(less_output))
        torch.cuda.empty_cache()
        
    y_more = np.mean(y_more, 0)
    y_less = np.mean(y_less, 0)
    overall_cv_score = get_score(y_more, y_less)
    logger.info(f'cv score {overall_cv_score}')
    
    oof_df = pd.DataFrame()
    oof_df['worker'] = df['worker']
    oof_df['more_oof'] = y_more
    oof_df['less_oof'] = y_less
    oof_df.to_csv(OUTPUT_DIR+"oof.csv", index=False)
    print(oof_df.shape)


def init_logger(log_file='train.log'):
    from logging import getLogger, INFO, FileHandler,  Formatter,  StreamHandler
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=log_file)
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger


logger = init_logger(log_file='log/' + f"{CFG.EXP_ID}.log")


# main loop
for fold in range(5):
    if fold not in CFG.folds:
        continue
    logger.info("=" * 120)
    logger.info(f"Fold {fold} Training")
    logger.info("=" * 120)

    trn_df = train[train.kfold != fold].reset_index(drop=True)
    val_df = train[train.kfold == fold].reset_index(drop=True)

    model = RoBERTaBase(CFG.model_name)    
  
    train_dataset = Jigsaw4Dataset(df=trn_df, cfg=CFG)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=CFG.train_bs, num_workers=0, pin_memory=True, shuffle=True
    )
    
    valid_dataset = Jigsaw4Dataset(df=val_df, cfg=CFG)
    valid_dataloader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=CFG.valid_bs, num_workers=0, pin_memory=True, shuffle=False
    )
    
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.001},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
    ]

    num_train_steps = int(len(trn_df) / CFG.train_bs * CFG.epochs)   
    optimizer = transformers.AdamW(optimizer_parameters, lr=CFG.LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=CFG.ETA_MIN, T_max=CFG.epochs)

    model = model.to(device)

    min_loss = 999
    best_score = np.inf

    for epoch in range(CFG.epochs):
        logger.info("Starting {} epoch...".format(epoch+1))

        start_time = time.time()

        train_avg, train_loss, valid_avg, valid_loss, best_score = train_fn(epoch, model, train_dataloader, valid_dataloader, device, optimizer, scheduler, best_score)
        # train_avg, train_loss = train_fn(model, train_dataloader, device, optimizer, scheduler)
        valid_avg, valid_loss = valid_fn(model, valid_dataloader, device)

        elapsed = time.time() - start_time
        
        logger.info(f'Epoch {epoch+1} - avg_train_loss: {train_loss:.5f}  avg_val_loss: {valid_loss:.5f}  time: {elapsed:.0f}s')
        logger.info(f"Epoch {epoch+1} - train_score:{train_avg['score']:0.5f}  valid_score:{valid_avg['score']:0.5f}")

        if valid_avg['score'] < best_score:
            logger.info(f">>>>>>>> Model Improved From {best_score} ----> {valid_avg['score']}")
            torch.save(model.state_dict(), OUTPUT_DIR+f'fold-{fold}.bin')
            best_score = valid_avg['score']


if len(CFG.folds) == 1:
    pass
else:
    model_paths = [OUTPUT_DIR+f'fold-{i}.bin' for i in CFG.folds]

    calc_cv(model_paths)
    print('calc cv finished!!')


