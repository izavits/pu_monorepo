import scipy.stats as stats
import pandas as pd
import numpy as np


def summarize_single_run(path, ratio):
    df = pd.read_csv(path)
    cols = ['run', 'pu_ratio', 'labeling_mechanism']
    cond = f'pu_ratio == {ratio}'
    df_ = df.copy()
    df_ = df_.query(cond)
    df_filtered = df_.drop(columns=cols)

    # Average Metrics
    print(f'Average scores for {cond}:')
    scores = df_filtered.copy().groupby('dataset').mean().mean(axis=0)
    print(df_filtered.copy().groupby('dataset').mean().mean(axis=0).to_string())
    print('')
    print(f'STD values for {cond}:')
    stds = df_filtered.copy().groupby('dataset').std().mean(axis=0)
    print(df_filtered.copy().groupby('dataset').std().mean(axis=0).to_string())
    return scores, stds


def summarize_multiple_runs(labeling, ratio):
    datasets = [
        'SCALED_CARDIO',
        'SCALED_CLIMATE',
        'SCALED_COVER',
        'SCALED_LETTER',
        'SCALED_MAMMOGRAPHY',
        'SCALED_PIECHART',
        'SCALED_PIZZACUTTER',
        'SCALED_POKER',
        'SCALED_SATELLITE',
        'SCALED_SEGMENT',
        'SCALED_SEISMIC',
        'SCALED_SHUTTLE',
        'SCALED_THYROID',
        'SCALED_YEAST'
    ]
    roc_per_dataset = []
    ap_per_dataset = []
    f1_per_dataset = []
    std_roc = []
    std_ap = []
    std_f1 = []
    for dataset in datasets:
        path=f'./output/{dataset}_{labeling}_{ratio}_iFPU_lgb/results.csv'
        scores, stds = summarize_single_run(path, ratio)
        roc_per_dataset.append(scores['roc_auc'])
        ap_per_dataset.append(scores['ap'])
        f1_per_dataset.append(scores['f1'])
        std_roc.append(stds['roc_auc'])
        std_ap.append(stds['ap'])
        std_f1.append(stds['f1'])
    # Average
    print('')
    print(f'Average scores (all datasets) for {labeling} and {ratio}:')
    print(f'roc_auc: {np.mean(roc_per_dataset)}')
    print(f'ap: {np.mean(ap_per_dataset)}')
    print(f'f1: {np.mean(f1_per_dataset)}')
    print(f'std_roc: {np.mean(std_roc)}')
    print(f'std_ap: {np.mean(std_ap)}')
    print(f'std_f1: {np.mean(std_f1)}')


def summarize_multiple_runs_sarem(labeling, ratio):
    datasets = [
        'SCALED_CARDIO',
        'SCALED_CLIMATE',
        'SCALED_COVER',
        'SCALED_LETTER',
        'SCALED_MAMMOGRAPHY',
        'SCALED_PIECHART',
        'SCALED_PIZZACUTTER',
        'SCALED_POKER',
        'SCALED_SATELLITE',
        'SCALED_SEGMENT',
        'SCALED_SEISMIC',
        'SCALED_SHUTTLE',
        'SCALED_THYROID',
        'SCALED_YEAST'
    ]
    roc_per_dataset = []
    ap_per_dataset = []
    f1_per_dataset = []
    std_roc = []
    std_ap = []
    std_f1 = []
    for dataset in datasets:
        path=f'./output_SAREM/{dataset}_{labeling}_{ratio}_default_SAREM/results.csv'
        scores, stds = summarize_single_run(path, ratio)
        roc_per_dataset.append(scores['roc_auc'])
        ap_per_dataset.append(scores['ap'])
        f1_per_dataset.append(scores['f1'])
        std_roc.append(stds['roc_auc'])
        std_ap.append(stds['ap'])
        std_f1.append(stds['f1'])
    # Average
    print('')
    print(f'Average scores (all datasets) for {labeling} and {ratio}:')
    print(f'roc_auc: {np.mean(roc_per_dataset)}')
    print(f'ap: {np.mean(ap_per_dataset)}')
    print(f'f1: {np.mean(f1_per_dataset)}')
    print(f'std_roc: {np.mean(std_roc)}')
    print(f'std_ap: {np.mean(std_ap)}')
    print(f'std_f1: {np.mean(std_f1)}')

def summarize_multiple_runs_lb3(labeling, ratio):
    datasets = [
        'SCALED_CARDIO',
        'SCALED_CLIMATE',
        'SCALED_COVER',
        'SCALED_LETTER',
        'SCALED_MAMMOGRAPHY',
        'SCALED_PIECHART',
        'SCALED_PIZZACUTTER',
        'SCALED_POKER',
        'SCALED_SATELLITE',
        'SCALED_SEGMENT',
        'SCALED_SEISMIC',
        'SCALED_SHUTTLE',
        'SCALED_THYROID',
        'SCALED_YEAST'
    ]
    roc_per_dataset = []
    ap_per_dataset = []
    f1_per_dataset = []
    std_roc = []
    std_ap = []
    std_f1 = []
    for dataset in datasets:
        path=f'./output_LBE/{dataset}_{labeling}_{ratio}_default_LBE/results.csv'
        scores, stds = summarize_single_run(path, ratio)
        roc_per_dataset.append(scores['roc_auc'])
        ap_per_dataset.append(scores['ap'])
        f1_per_dataset.append(scores['f1'])
        std_roc.append(stds['roc_auc'])
        std_ap.append(stds['ap'])
        std_f1.append(stds['f1'])
    # Average
    print('')
    print(f'Average scores (all datasets) for {labeling} and {ratio}:')
    print(f'roc_auc: {np.mean(roc_per_dataset)}')
    print(f'ap: {np.mean(ap_per_dataset)}')
    print(f'f1: {np.mean(f1_per_dataset)}')
    print(f'std_roc: {np.mean(std_roc)}')
    print(f'std_ap: {np.mean(std_ap)}')
    print(f'std_f1: {np.mean(std_f1)}')


def summarize_multiple_runs_negativity_assumption(labeling, ratio):
    datasets = [
        'SCALED_CARDIO',
        'SCALED_CLIMATE',
        'SCALED_COVER',
        'SCALED_LETTER',
        'SCALED_MAMMOGRAPHY',
        'SCALED_PIECHART',
        'SCALED_PIZZACUTTER',
        'SCALED_POKER',
        'SCALED_SATELLITE',
        'SCALED_SEGMENT',
        'SCALED_SEISMIC',
        'SCALED_SHUTTLE',
        'SCALED_THYROID',
        'SCALED_YEAST'
    ]
    roc_per_dataset = []
    ap_per_dataset = []
    f1_per_dataset = []
    std_roc = []
    std_ap = []
    std_f1 = []
    for dataset in datasets:
        path=f'./output_PN_noise/{dataset}_{labeling}_{ratio}_FL_lgb/results.csv'
        scores, stds = summarize_single_run(path, ratio)
        roc_per_dataset.append(scores['roc_auc'])
        ap_per_dataset.append(scores['ap'])
        f1_per_dataset.append(scores['f1'])
        std_roc.append(stds['roc_auc'])
        std_ap.append(stds['ap'])
        std_f1.append(stds['f1'])
    # Average
    print('')
    print(f'Average scores (all datasets) for {labeling} and {ratio}:')
    print(f'roc_auc: {np.mean(roc_per_dataset)}')
    print(f'ap: {np.mean(ap_per_dataset)}')
    print(f'f1: {np.mean(f1_per_dataset)}')
    print(f'std_roc: {np.mean(std_roc)}')
    print(f'std_ap: {np.mean(std_ap)}')
    print(f'std_f1: {np.mean(std_f1)}')

def summarize_mutiple_runs_2step(path, labeling, ratio, ad_method, prior=None):
    datasets = [
        'SCALED_CARDIO',
        'SCALED_CLIMATE',
        'SCALED_COVER',
        'SCALED_LETTER',
        'SCALED_MAMMOGRAPHY',
        'SCALED_PIECHART',
        'SCALED_PIZZACUTTER',
        'SCALED_POKER',
        'SCALED_SATELLITE',
        'SCALED_SEGMENT',
        'SCALED_SEISMIC',
        'SCALED_SHUTTLE',
        'SCALED_THYROID',
        'SCALED_YEAST'
    ]
    roc_per_dataset = []
    ap_per_dataset = []
    f1_per_dataset = []
    std_roc = []
    std_ap = []
    std_f1 = []
    for dataset in datasets:
        if prior:
            filepath = f'./{path}/{dataset}_{labeling}_{ratio}_{ad_method}_iFPNU_lgb_{prior}/results.csv'
        else:
            filepath = f'./{path}/{dataset}_{labeling}_{ratio}_{ad_method}_iFPNU_lgb/results.csv'
        scores, stds = summarize_single_run(filepath, ratio)
        roc_per_dataset.append(scores['roc_auc'])
        ap_per_dataset.append(scores['ap'])
        f1_per_dataset.append(scores['f1'])
        std_roc.append(stds['roc_auc'])
        std_ap.append(stds['ap'])
        std_f1.append(stds['f1'])
    # Average
    print('')
    print(f'Average scores (all datasets) for {labeling} and {ratio}:')
    print(f'roc_auc: {np.mean(roc_per_dataset)}')
    print(f'ap: {np.mean(ap_per_dataset)}')
    print(f'f1: {np.mean(f1_per_dataset)}')
    print(f'std_roc: {np.mean(std_roc)}')
    print(f'std_ap: {np.mean(std_ap)}')
    print(f'std_f1: {np.mean(std_f1)}')

    iFPNU = {
        'roc_auc': roc_per_dataset,
        'ap': ap_per_dataset,
        'f1': f1_per_dataset
    }
    print(iFPNU)

if __name__ == '__main__':
    #summarize_single_run('./output/SCALED_COVER_SAR_0.75_iFPU_lgb/results.csv', 0.75)
    #summarize_multiple_runs('SAR', 0.75)
    summarize_mutiple_runs_2step('output_2step', 'SCAR', 0.25, 'dif', None)
    #summarize_multiple_runs_sarem('SAR', 0.75)
    #summarize_multiple_runs_lb3('SAR', 0.75)
    #summarize_multiple_runs_negativity_assumption('SAR', 0.75)