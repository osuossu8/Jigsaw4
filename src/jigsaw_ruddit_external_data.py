import numpy as np
import pandas as pd
import os
import gc
import sys
sys.path.append("/root/workspace/Jigsaw4")
from tqdm import tqdm

# https://www.kaggle.com/rajkumarl/ruddit-jigsaw-dataset
ruddit_path = 'input/RudditJigsaw/Dataset/'

df = pd.read_csv(ruddit_path+'ruddit_with_text.csv')
df = df[['txt', 'offensiveness_score']].rename(columns = {'txt': 'text', 'offensiveness_score': 'y'})
df['y'] = (df['y'] - df.y.min()) / (df.y.max() - df.y.min())

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

df.to_csv(ruddit_path+'jigsaw_ruddit_toxic_score.csv', index=False)

