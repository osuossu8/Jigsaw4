import numpy as np
import pandas as pd
import os
import gc
import sys
sys.path.append("/root/workspace/Jigsaw4")
from tqdm import tqdm

toxic_comment_path = 'input/JigsawToxicComment/'

df_test = pd.read_csv(toxic_comment_path+'test.csv')

df_test_label = pd.read_csv(toxic_comment_path+'test_labels.csv').replace(-1,0)

df_test = pd.merge(df_test, df_test_label, how="left", on = "id")
print(df_test.shape)

df_train = pd.read_csv(toxic_comment_path+'train.csv')

df = pd.concat([df_train, df_test])
print(df.shape)

del df_train, df_test, df_test_label

# df['severe_toxic'] = df.severe_toxic * 2
df['y'] = (df[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].sum(axis=1) ).astype(int)
df['y'] = df['y']/df['y'].max()

df = df[['comment_text', 'y']].rename(columns={'comment_text': 'text'})
print(df.sample(5))


# Validation data 

df_val = pd.read_csv("input/validation_data.csv")
print(df_val.shape)


# Find cases already present in toxic data

df_val = pd.merge(df_val, df.loc[:,['text', 'y']], 
                  left_on = 'less_toxic', 
                  right_on = 'text', how='left')

df_val = pd.merge(df_val, df.loc[:,['text', 'y']], 
                  left_on = 'more_toxic', 
                  right_on = 'text', how='left')

print(df_val.head())

# Removing those cases
df_val = df_val[(~df_val.y_x.isna()) & (~df_val.y_y.isna())]
df_more = df_val[['more_toxic', 'y_y']]
df_less = df_val[['less_toxic', 'y_x']]
df_more.columns = ['text', 'y']
df_less.columns = ['text', 'y']
df_val = pd.concat([df_more, df_less], 0).reset_index(drop=True)
df_val = df_val.groupby('text')['y'].mean().reset_index()
print(df_val.shape)
print(df_val.head())

print(df.shape)
df_ = pd.merge(df.reset_index(), df_val.reset_index(), on=['text', 'y'], how='left')
df_ = df_.dropna()

df = df.reset_index()
df = df[~df['index'].isin(df_.index)].reset_index(drop=True)

del df['index']
print(df.shape)
print(df['y'].value_counts())
print(df.head())

df.to_csv(toxic_comment_path+'jigsaw_toxic_comment_toxic_score_no_weighted.csv', index=False)

