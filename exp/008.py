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
    EXP_ID = '008'
    seed = 71
    epochs = 3
    folds = [0, 1, 2, 3, 4]
    N_FOLDS = 5
    LR = 1e-4 # 2e-5
    ETA_MIN = 1e-7 # 1e-5
    T_MAX = 500
    max_len = 128 # 256
    train_bs = 64 # 16 * 2
    valid_bs = 128 # 32 * 2
    log_interval = 50
    model_name = 'roberta-base'
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
        self.more_pred = []
        self.less_pred = []
    
    def update(self, more_pred, less_pred):
        self.more_pred.extend(more_pred.cpu().detach().numpy().tolist())
        self.less_pred.extend(less_pred.cpu().detach().numpy().tolist())

    @property
    def avg(self):
        self.score = get_score(self.more_pred, self.less_pred)
       
        return {
            "score" : self.score,
        }

# ====================================================
# Data Loading
# ====================================================
train = pd.read_csv('input/validation_data.csv')
if CFG.DEBUG:
    train = train.sample(n=100, random_state=CFG.seed).reset_index(drop=True)
test = pd.read_csv('input/comments_to_score.csv')
submission = pd.read_csv('input/sample_submission.csv')
print(train.shape)
print(test.shape, submission.shape)


# ====================================================
# CV split
# ====================================================
Fold = GroupKFold(n_splits=CFG.N_FOLDS)
for n, (trn_index, val_index) in enumerate(Fold.split(train, train, train['worker'])):
    train.loc[val_index, 'kfold'] = int(n)
train['kfold'] = train['kfold'].astype(int)


train.to_csv('input/train_folds.csv', index=False)


class Jigsaw4Dataset:
    def __init__(self, df, cfg):
        self.tokenizer = cfg.tokenizer
        self.max_len = cfg.max_len
        self.less_toxic = df['less_toxic'].fillna("none").values
        self.more_toxic = df['more_toxic'].fillna("none").values

    def __len__(self):
        return len(self.less_toxic)

    def __getitem__(self, item):

        less_toxic_inputs = self.tokenizer(
            str(self.less_toxic[item]), 
            max_length=self.max_len, 
            padding="max_length", 
            truncation=True
        )
        
        more_toxic_inputs = self.tokenizer(
            str(self.more_toxic[item]), 
            max_length=self.max_len, 
            padding="max_length", 
            truncation=True
        )

        less_toxic_ids = less_toxic_inputs["input_ids"]
        less_toxic_mask = less_toxic_inputs["attention_mask"]
        
        more_toxic_ids = more_toxic_inputs["input_ids"]
        more_toxic_mask = more_toxic_inputs["attention_mask"]
        
        targets = torch.tensor(1, dtype=torch.float)

        return {
            "less_input_ids": torch.tensor(less_toxic_ids, dtype=torch.long),
            "less_attention_mask": torch.tensor(less_toxic_mask, dtype=torch.long),
            "more_input_ids": torch.tensor(more_toxic_ids, dtype=torch.long),
            "more_attention_mask": torch.tensor(more_toxic_mask, dtype=torch.long),
            "targets" : torch.tensor(1, dtype=torch.float32),
        }


class AttentionHead(nn.Module):
    def __init__(self, in_features, hidden_dim, num_targets):
        super().__init__()
        self.in_features = in_features
        self.middle_features = hidden_dim
        self.W = nn.Linear(in_features, hidden_dim)
        self.V = nn.Linear(hidden_dim, 1)
        self.out_features = hidden_dim

    def forward(self, features):
        att = torch.tanh(self.W(features))
        score = self.V(att)
        attention_weights = torch.softmax(score, dim=1)
        context_vector = attention_weights * features
        context_vector = torch.sum(context_vector, dim=1)

        return context_vector


class RoBERTaBase(nn.Module):
    def __init__(self, model_path):
        super(RoBERTaBase, self).__init__()
        self.in_features = 768
        self.roberta = AutoModel.from_pretrained(model_path, return_dict=False)
        self.layer_norm = nn.LayerNorm(768)
        self.dropout = nn.Dropout(0.2)
        self.dense = nn.Sequential(
            nn.Linear(768, 256),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(0.2),
            nn.Linear(256, 1)
        )

    def forward(self, ids, mask):
        _, pooled_output = self.roberta(
            ids,
            attention_mask=mask
        )
        pooled_output = self.layer_norm(pooled_output)
        pooled_output = self.dropout(pooled_output)
        logits = self.dense(pooled_output)
        return logits.squeeze(-1)


def loss_fn(more_toxic_logits, less_toxic_logits, targets):
    loss_fct = nn.MarginRankingLoss(margin=CFG.margin)
    loss = loss_fct(more_toxic_logits, less_toxic_logits, targets)
    return loss


def train_fn(epoch, model, train_data_loader, valid_data_loader, device, optimizer, scheduler, best_score):
    model.train()
    losses = AverageMeter()
    scores = MetricMeter()
    tk0 = tqdm(train_data_loader, total=len(train_data_loader))
    
    for batch_idx, data in enumerate(tk0):
        optimizer.zero_grad()
        less_inputs = data['less_input_ids'].to(device)
        less_masks = data['less_attention_mask'].to(device)
        more_inputs = data['more_input_ids'].to(device)
        more_masks = data['more_attention_mask'].to(device)
        targets = data['targets'].to(device)
        
        less_toxic_logits = model(less_inputs, less_masks)
        more_toxic_logits = model(more_inputs, more_masks)

        loss = loss_fn(more_toxic_logits, less_toxic_logits, targets)
        loss.backward()
        optimizer.step()
        scheduler.step()
        losses.update(loss.item(), less_inputs.size(0))
        scores.update(more_toxic_logits, less_toxic_logits)
        tk0.set_postfix(loss=losses.avg)

        if (batch_idx > 0) and (batch_idx % CFG.log_interval == 0):
            valid_avg, valid_loss = valid_fn(model, valid_data_loader, device)

            logger.info(f"Epoch {epoch+1}, Step {batch_idx} - valid_score:{valid_avg['score']:0.5f}")

            if valid_avg['score'] > best_score:
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
            less_inputs = data['less_input_ids'].to(device)
            less_masks = data['less_attention_mask'].to(device)
            more_inputs = data['more_input_ids'].to(device)
            more_masks = data['more_attention_mask'].to(device)
            targets = data['targets'].to(device)
   
            less_toxic_logits = model(less_inputs, less_masks)
            more_toxic_logits = model(more_inputs, more_masks)

            loss = loss_fn(more_toxic_logits, less_toxic_logits, targets)
            losses.update(loss.item(), less_inputs.size(0))
            scores.update(more_toxic_logits, less_toxic_logits)
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

    y_more = []
    y_less = []
    idx = []
    for fold, model in enumerate(models):
        val_df = df[df.kfold == fold].reset_index(drop=True)
    
        dataset = Jigsaw4Dataset(df=val_df, cfg=CFG)
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=CFG.valid_bs, num_workers=0, pin_memory=True, shuffle=False
        )

        more_output = []
        less_output = []
        for b_idx, data in tqdm(enumerate(data_loader)):
            with torch.no_grad():
                less_inputs = data['less_input_ids'].to(device)
                less_masks = data['less_attention_mask'].to(device)
                more_inputs = data['more_input_ids'].to(device)
                more_masks = data['more_attention_mask'].to(device)

                less_toxic_logits = model(less_inputs, less_masks)
                more_toxic_logits = model(more_inputs, more_masks)
                
                more_toxic_logits = more_toxic_logits.detach().cpu().numpy().tolist()
                less_toxic_logits = less_toxic_logits.detach().cpu().numpy().tolist()
                more_output.extend(more_toxic_logits)
                less_output.extend(less_toxic_logits)
        logger.info(get_score(np.array(more_toxic_logits), np.array(less_toxic_logits)))
        y_more.append(np.array(more_output))
        y_less.append(np.array(less_output))
        idx.append(val_df['worker'].values)
        torch.cuda.empty_cache()
        
    y_more = np.concatenate(y_more)
    y_less = np.concatenate(y_less)
    idx = np.concatenate(idx)
    overall_cv_score = get_score(y_more, y_less)
    logger.info(f'cv score {overall_cv_score}')
    
    oof_df = pd.DataFrame()
    oof_df['worker'] = idx
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
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=CFG.ETA_MIN, T_max=CFG.T_MAX)

    model = model.to(device)

    min_loss = 999
    best_score = 0. # np.inf

    for epoch in range(CFG.epochs):
        logger.info("Starting {} epoch...".format(epoch+1))

        start_time = time.time()

        train_avg, train_loss, valid_avg, valid_loss, best_score = train_fn(epoch, model, train_dataloader, valid_dataloader, device, optimizer, scheduler, best_score)
        # train_avg, train_loss = train_fn(model, train_dataloader, device, optimizer, scheduler)
        valid_avg, valid_loss = valid_fn(model, valid_dataloader, device)

        elapsed = time.time() - start_time
        
        logger.info(f'Epoch {epoch+1} - avg_train_loss: {train_loss:.5f}  avg_val_loss: {valid_loss:.5f}  time: {elapsed:.0f}s')
        logger.info(f"Epoch {epoch+1} - train_score:{train_avg['score']:0.5f}  valid_score:{valid_avg['score']:0.5f}")

        if valid_avg['score'] > best_score:
            logger.info(f">>>>>>>> Model Improved From {best_score} ----> {valid_avg['score']}")
            torch.save(model.state_dict(), OUTPUT_DIR+f'fold-{fold}.bin')
            best_score = valid_avg['score']


if len(CFG.folds) == 1:
    pass
else:
    model_paths = [OUTPUT_DIR+f'fold-{i}.bin' for i in CFG.folds]

    calc_cv(model_paths)
    print('calc cv finished!!')


