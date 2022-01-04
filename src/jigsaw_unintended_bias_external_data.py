import numpy as np
import pandas as pd
import os
import gc
import sys
sys.path.append("/root/workspace/Jigsaw4")
from tqdm import tqdm


jigsaw_unintended = pd.read_csv(
    'input/JigsawUnintendedBias/all_data.csv',
    usecols=['id', 'comment_text', 'toxicity', 'severe_toxicity', 'obscene', 
            'insult', 'threat', 'identity_attack']
)
print(jigsaw_unintended.shape)


# Give more weight to severe toxic 
jigsaw_unintended['severe_toxicity'] = jigsaw_unintended.severe_toxicity * 2
jigsaw_unintended['y'] = (jigsaw_unintended[['toxicity', 'severe_toxicity', 'obscene', 'insult', 'threat', 'identity_attack']].sum(axis=1) ).astype(float)
jigsaw_unintended['y'] = jigsaw_unintended['y']/jigsaw_unintended['y'].max()

jigsaw_unintended = jigsaw_unintended[['comment_text', 'y']].rename(columns={'comment_text': 'text'})


# Validation data 

df_val = pd.read_csv("input/validation_data.csv")
print(df_val.shape)


# Find cases already present in toxic data

df_val = pd.merge(df_val, jigsaw_unintended.loc[:,['text']], 
                  left_on = 'less_toxic', 
                  right_on = 'text', how='left')

df_val = pd.merge(df_val, jigsaw_unintended.loc[:,['text']], 
                  left_on = 'more_toxic', 
                  right_on = 'text', how='left')

# Removing those cases
df_val = df_val[(~df_val.text_x.isna()) | (~df_val.text_y.isna())][['worker', 'less_toxic', 'more_toxic']]
print(df_val.shape)


jigsaw_unintended.to_csv('input/JigsawUnintendedBias/jigsaw_unintended_bias_toxic_score.csv', index=False)

