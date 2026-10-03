# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 21:17:46 2020

@author: orteg
"""
import scipy.stats as stats
import pandas as pd
import numpy as np

# Load Data
df = pd.read_csv('output/results_cardio.csv')
cols = ['data_partition', 'flip_ratio', 'label_noise']
cond = 'flip_ratio == 0.25'

df_ = df.copy()
df_ = df_.query(cond)
df_filtered = df_.drop(columns=cols)

# Average Metrics
print(f'Average scores for {cond}:')
print(df_filtered.copy().groupby('dataset').mean().mean(axis=0).to_string())
print('')
print(f'STD values for {cond}:')
print(df_filtered.copy().groupby('dataset').std().mean(axis=0).to_string())

#### TECHNIQUES ####
roc_models = ['roc_srf_puhd', 'roc_dt_puhd']
ap_models = ['ap_srf_puhd', 'ap_dt_puhd']
f1max_models = ['f1_max_srf_puhd', 'f1_max_dt_puhd']

## Evaluation ##
# Optimal F1-score
df_f1scoremax_ = df_[f1max_models].copy().rank(axis=1, ascending=False)
df_f1scoremax_['dataset'] = df_['dataset']

df_f1_max_transpose = df_f1scoremax_.groupby('dataset').mean().T
df_f1scoremax_rank_mean_ = df_f1_max_transpose.mean(axis=1)

# PR-AUC
df_ap_ = df_[ap_models].copy().rank(axis=1, ascending=False)
df_ap_['dataset'] = df_['dataset']

df_ap_transpose = df_ap_.groupby('dataset').mean().T
df_ap_rank_mean_ = df_ap_transpose.mean(axis=1)

# ROC-AUC 
df_roc_ = df_[roc_models].copy().rank(axis=1, ascending=False)
df_roc_['dataset'] = df_['dataset']

df_roc_transpose = df_roc_.groupby('dataset').mean().T
df_roc_rank_mean_ = df_roc_transpose.mean(axis=1)
