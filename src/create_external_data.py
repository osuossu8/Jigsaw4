import numpy as np
import pandas as pd
import os
import gc
import sys
sys.path.append("/root/workspace/Jigsaw4")
from tqdm import tqdm


past = pd.read_csv(
    'input/JigsawUnintendedBias/all_data.csv',
    usecols=['id', 'comment_text', 'toxicity', 'severe_toxicity', 'obscene', 
            'insult', 'threat', 'toxicity_annotator_count']
    )

toxic_annotation = pd.read_csv(
    'input/JigsawUnintendedBias/toxicity_individual_annotations.csv',
    usecols=['id', 'worker', 'toxic', 'severe_toxic', 'insult', 'obscene', 'threat']
)

toxic_annotation = pd.merge(toxic_annotation, past[['id', 'comment_text']], on='id')
print(toxic_annotation.shape)

severe_workers = toxic_annotation.query('toxic == 1 and severe_toxic == 1')['worker'].unique()
toxic_workers = toxic_annotation.query('toxic == 1 and severe_toxic == 0')['worker'].unique()
intersec = list(set(severe_workers) & set(toxic_workers))

res = []
for i in tqdm(intersec):
    more_toxic = toxic_annotation.query(f'worker == {i}').query('toxic == 1 and severe_toxic == 1').reset_index(drop=True)['comment_text']
    len_more = len(more_toxic)
    less_toxic = toxic_annotation.query(f'worker == {i}').query('toxic == 1 and severe_toxic == 0').reset_index(drop=True)['comment_text'].head(len_more)
    tmp = pd.DataFrame()
    tmp['more_toxic'] = more_toxic
    tmp['less_toxic'] = less_toxic
    tmp['worker'] = i
    res.append(tmp)
    
res = pd.concat(res)
print(res.shape)
print(res.head())

res.to_csv('input/JigsawUnintendedBias/external_data_jigsaw_unintended_bias.csv', index=False)

