import pandas as pd
import numpy as np
from scipy.stats import friedmanchisquare
import scikit_posthocs as sp

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

MECHANISMS = ['SCAR', 'SAR']
# F1 at the 0.5 threshold for the positive class, the negative class and their macro average
F1_05_METRICS = ["f1_pos_05", "f1_neg_05", "f1_macro_05"]
# F1 when the top pi (class prior) fraction of the test scores is predicted positive (only in this project)
F1_PRIOR_METRICS = ["f1_pos_prior", "f1_neg_prior", "f1_macro_prior"]
# PU trees: their F1 at the class-prior threshold (pos, neg, macro). NOTE: for now these use prob > prior
# (prior of the whole dataset), not the top-prior-fraction rule used in this project and in ALM-PU
DT_F1_PRIOR_COLS = ('f1_ci_pos_dt_puhd', 'f1_ci_neg_dt_puhd', 'f1_ci_macro_dt_puhd')  # decision tree
SRF_F1_PRIOR_COLS = ('f1_ci_pos_srf_puhd', 'f1_ci_neg_srf_puhd', 'f1_ci_macro_srf_puhd')  # random forest
METRICS = ["roc_auc", "ap", "f1", "std_roc", "std_ap", "std_f1"] + F1_05_METRICS + \
          [f'std_{m}' for m in F1_05_METRICS] + F1_PRIOR_METRICS + [f'std_{m}' for m in F1_PRIOR_METRICS]
# Column names of the F1@0.5 metrics (pos, neg, macro) in the results of each project
F1_05_COLS = ('f1_pos_05', 'f1_neg_05', 'f1_macro_05')  # this project
ALMPU_F1_05_COLS = ('f1_pos', 'f1_neg', 'f1_macro')
DT_F1_05_COLS = ('f1_pos_dt_puhd', 'f1_neg_dt_puhd', 'f1_macro_dt_puhd')  # PU trees: decision tree
SRF_F1_05_COLS = ('f1_pos_srf_puhd', 'f1_neg_srf_puhd', 'f1_macro_srf_puhd')  # PU trees: random forest
PU_RATIOS = [0.25, 0.5, 0.75]
METHODS = ['iFPU', 'dif_iFPNU', 'eif_iFPNU', 'optiforest_iFPNU',
           'LBE', 'FL_lgb', 'SAREM', 'PUe', 'uPU', 'nnPU', 'imbNNPU', 'PUHDT', 'PUHRF', 'DIF', 'ALMPU']


def empty_results():
    return {
        mech: {ratio: {m: [] for m in METRICS} for ratio in PU_RATIOS}
        for mech in MECHANISMS
    }


results = {method: empty_results() for method in METHODS}


def parse_result_csv(path):
    df = pd.read_csv(path)
    mean_roc_auc = df["roc_auc"].mean()
    mean_ap = df["ap"].mean()
    mean_f1 = df["f1"].mean()
    std_roc = df["roc_auc"].std()
    std_ap = df["ap"].std()
    std_f1 = df["f1"].std()
    return mean_roc_auc, mean_ap, mean_f1, std_roc, std_ap, std_f1


def add_f1_05(method, label, ratio, df, columns):
    """
    Store the mean and std of the F1@0.5 metrics (pos, neg, macro) of one dataset
    :param df: the runs (or data partitions) of the dataset
    :param columns: the names of the F1@0.5 pos, neg and macro columns in df
    """
    for metric, col in zip(F1_05_METRICS, columns):
        results[method][label][ratio][metric].append(df[col].mean())
        results[method][label][ratio][f'std_{metric}'].append(df[col].std())


def add_f1_prior(method, label, ratio, df, columns=tuple(F1_PRIOR_METRICS)):
    """
    Store the mean and std of the F1 metrics (pos, neg, macro) at the class-prior threshold of one dataset.
    Results from runs before these metrics existed get NaN; produce_tables() then leaves the method out.
    :param df: the runs (or data partitions) of the dataset
    :param columns: the names of the pos, neg and macro columns in df
    """
    for metric, col in zip(F1_PRIOR_METRICS, columns):
        has_metric = col in df.columns
        results[method][label][ratio][metric].append(df[col].mean() if has_metric else np.nan)
        results[method][label][ratio][f'std_{metric}'].append(df[col].std() if has_metric else np.nan)


def gather_results():
    for label in MECHANISMS:
        for ratio in PU_RATIOS:
            for data in datasets:
                #iFPU
                path = f'./output/{data}_{label}_{ratio}_iFPU_lgb/results.csv'
                roc, ap, f1, std_roc, std_ap, std_f1 = parse_result_csv(path)
                results['iFPU'][label][ratio]['roc_auc'].append(roc)
                results['iFPU'][label][ratio]['ap'].append(ap)
                results['iFPU'][label][ratio]['f1'].append(f1)
                results['iFPU'][label][ratio]['std_roc'].append(std_roc)
                results['iFPU'][label][ratio]['std_ap'].append(std_ap)
                results['iFPU'][label][ratio]['std_f1'].append(std_f1)
                add_f1_prior('iFPU', label, ratio, pd.read_csv(path))
                add_f1_05('iFPU', label, ratio, pd.read_csv(path), F1_05_COLS)
                #DIF_iFPNU
                path = f'./output_2step/{data}_{label}_{ratio}_dif_iFPNU_lgb/results.csv'
                roc, ap, f1, std_roc, std_ap, std_f1 = parse_result_csv(path)
                results['dif_iFPNU'][label][ratio]['roc_auc'].append(roc)
                results['dif_iFPNU'][label][ratio]['ap'].append(ap)
                results['dif_iFPNU'][label][ratio]['f1'].append(f1)
                results['dif_iFPNU'][label][ratio]['std_roc'].append(std_roc)
                results['dif_iFPNU'][label][ratio]['std_ap'].append(std_ap)
                results['dif_iFPNU'][label][ratio]['std_f1'].append(std_f1)
                add_f1_prior('dif_iFPNU', label, ratio, pd.read_csv(path))
                add_f1_05('dif_iFPNU', label, ratio, pd.read_csv(path), F1_05_COLS)
                #EIF_iFPNU
                path = f'./output_2step/{data}_{label}_{ratio}_eif_iFPNU_lgb/results.csv'
                roc, ap, f1, std_roc, std_ap, std_f1 = parse_result_csv(path)
                results['eif_iFPNU'][label][ratio]['roc_auc'].append(roc)
                results['eif_iFPNU'][label][ratio]['ap'].append(ap)
                results['eif_iFPNU'][label][ratio]['f1'].append(f1)
                results['eif_iFPNU'][label][ratio]['std_roc'].append(std_roc)
                results['eif_iFPNU'][label][ratio]['std_ap'].append(std_ap)
                results['eif_iFPNU'][label][ratio]['std_f1'].append(std_f1)
                add_f1_prior('eif_iFPNU', label, ratio, pd.read_csv(path))
                add_f1_05('eif_iFPNU', label, ratio, pd.read_csv(path), F1_05_COLS)
                #OptIforest_iFPNU
                path = f'./output_2step/{data}_{label}_{ratio}_optiforest_iFPNU_lgb/results.csv'
                roc, ap, f1, std_roc, std_ap, std_f1 = parse_result_csv(path)
                results['optiforest_iFPNU'][label][ratio]['roc_auc'].append(roc)
                results['optiforest_iFPNU'][label][ratio]['ap'].append(ap)
                results['optiforest_iFPNU'][label][ratio]['f1'].append(f1)
                results['optiforest_iFPNU'][label][ratio]['std_roc'].append(std_roc)
                results['optiforest_iFPNU'][label][ratio]['std_ap'].append(std_ap)
                results['optiforest_iFPNU'][label][ratio]['std_f1'].append(std_f1)
                add_f1_prior('optiforest_iFPNU', label, ratio, pd.read_csv(path))
                #add_f1_05('optiforest_iFPNU', label, ratio, pd.read_csv(path), F1_05_COLS)
                #LBE
                path = f'./output_LBE/{data}_{label}_{ratio}_default_LBE/results.csv'
                roc, ap, f1, std_roc, std_ap, std_f1 = parse_result_csv(path)
                results['LBE'][label][ratio]['roc_auc'].append(roc)
                results['LBE'][label][ratio]['ap'].append(ap)
                results['LBE'][label][ratio]['f1'].append(f1)
                results['LBE'][label][ratio]['std_roc'].append(std_roc)
                results['LBE'][label][ratio]['std_ap'].append(std_ap)
                results['LBE'][label][ratio]['std_f1'].append(std_f1)
                add_f1_prior('LBE', label, ratio, pd.read_csv(path))
                add_f1_05('LBE', label, ratio, pd.read_csv(path), F1_05_COLS)
                #FL_lgb (Negativity assumption)
                path = f'./output_PN_noise/{data}_{label}_{ratio}_FL_lgb/results.csv'
                roc, ap, f1, std_roc, std_ap, std_f1 = parse_result_csv(path)
                results['FL_lgb'][label][ratio]['roc_auc'].append(roc)
                results['FL_lgb'][label][ratio]['ap'].append(ap)
                results['FL_lgb'][label][ratio]['f1'].append(f1)
                results['FL_lgb'][label][ratio]['std_roc'].append(std_roc)
                results['FL_lgb'][label][ratio]['std_ap'].append(std_ap)
                results['FL_lgb'][label][ratio]['std_f1'].append(std_f1)
                #SAREM
                path = f'./output_SAREM/{data}_{label}_{ratio}_default_SAREM/results.csv'
                roc, ap, f1, std_roc, std_ap, std_f1 = parse_result_csv(path)
                results['SAREM'][label][ratio]['roc_auc'].append(roc)
                results['SAREM'][label][ratio]['ap'].append(ap)
                results['SAREM'][label][ratio]['f1'].append(f1)
                results['SAREM'][label][ratio]['std_roc'].append(std_roc)
                results['SAREM'][label][ratio]['std_ap'].append(std_ap)
                results['SAREM'][label][ratio]['std_f1'].append(std_f1)
                add_f1_prior('SAREM', label, ratio, pd.read_csv(path))
                add_f1_05('SAREM', label, ratio, pd.read_csv(path), F1_05_COLS)
                #PUe
                path = f'./output_PUe/{data}_{label}_{ratio}_default_PUe/results.csv'
                roc, ap, f1, std_roc, std_ap, std_f1 = parse_result_csv(path)
                results['PUe'][label][ratio]['roc_auc'].append(roc)
                results['PUe'][label][ratio]['ap'].append(ap)
                results['PUe'][label][ratio]['f1'].append(f1)
                results['PUe'][label][ratio]['std_roc'].append(std_roc)
                results['PUe'][label][ratio]['std_ap'].append(std_ap)
                results['PUe'][label][ratio]['std_f1'].append(std_f1)
                add_f1_prior('PUe', label, ratio, pd.read_csv(path))
                add_f1_05('PUe', label, ratio, pd.read_csv(path), F1_05_COLS)
                # DIF ablation (only AD step)
                path = f'./output_DIF/{data}_{label}_{ratio}_default_DIF/results.csv'
                roc, ap, f1, std_roc, std_ap, std_f1 = parse_result_csv(path)
                results['DIF'][label][ratio]['roc_auc'].append(roc)
                results['DIF'][label][ratio]['ap'].append(ap)
                results['DIF'][label][ratio]['f1'].append(f1)
                results['DIF'][label][ratio]['std_roc'].append(std_roc)
                results['DIF'][label][ratio]['std_ap'].append(std_ap)
                results['DIF'][label][ratio]['std_f1'].append(std_f1)
                # ALM-PU (run in the ALM-PU project with the same splits as main_LBE.py)
                path = f'./output_ALMPU/{data}_{label}_{ratio}_default_ALMPU/results.csv'
                roc, ap, f1, std_roc, std_ap, std_f1 = parse_result_csv(path)
                results['ALMPU'][label][ratio]['roc_auc'].append(roc)
                results['ALMPU'][label][ratio]['ap'].append(ap)
                results['ALMPU'][label][ratio]['f1'].append(f1)
                results['ALMPU'][label][ratio]['std_roc'].append(std_roc)
                results['ALMPU'][label][ratio]['std_ap'].append(std_ap)
                results['ALMPU'][label][ratio]['std_f1'].append(std_f1)
                add_f1_prior('ALMPU', label, ratio, pd.read_csv(path))
                add_f1_05('ALMPU', label, ratio, pd.read_csv(path), ALMPU_F1_05_COLS)


def gather_hellinger_trees_results():
    data = ['cardio', 'climate', 'cover', 'letter', 'mammography', 'piechart', 'pizzacutter', 'poker',
            'satellite', 'segment', 'seismic', 'shuttle', 'thyroid', 'yeast']
    # SCAR
    for d in data:
        path = f'./pu_trees_results/output_hellinger/results_{d}.csv'
        df = pd.read_csv(path)
        for r, grp in df.groupby(["flip_ratio"]):
            ratio = r[0]
            # PU-HDT
            results['PUHDT']['SCAR'][ratio]['roc_auc'].append(grp['roc_dt_puhd'].mean())
            results['PUHDT']['SCAR'][ratio]['ap'].append(grp['ap_dt_puhd'].mean())
            results['PUHDT']['SCAR'][ratio]['f1'].append(grp['f1_max_dt_puhd'].mean())
            results['PUHDT']['SCAR'][ratio]['std_roc'].append(grp['roc_dt_puhd'].std())
            results['PUHDT']['SCAR'][ratio]['std_ap'].append(grp['ap_dt_puhd'].std())
            results['PUHDT']['SCAR'][ratio]['std_f1'].append(grp['f1_max_dt_puhd'].std())
            add_f1_05('PUHDT', 'SCAR', ratio, grp, DT_F1_05_COLS)
            add_f1_prior('PUHDT', 'SCAR', ratio, grp, DT_F1_PRIOR_COLS)
            # PU-HRF
            results['PUHRF']['SCAR'][ratio]['roc_auc'].append(grp['roc_srf_puhd'].mean())
            results['PUHRF']['SCAR'][ratio]['ap'].append(grp['ap_srf_puhd'].mean())
            results['PUHRF']['SCAR'][ratio]['f1'].append(grp['f1_max_srf_puhd'].mean())
            results['PUHRF']['SCAR'][ratio]['std_roc'].append(grp['roc_srf_puhd'].std())
            results['PUHRF']['SCAR'][ratio]['std_ap'].append(grp['ap_srf_puhd'].std())
            results['PUHRF']['SCAR'][ratio]['std_f1'].append(grp['f1_max_srf_puhd'].std())
            add_f1_05('PUHRF', 'SCAR', ratio, grp, SRF_F1_05_COLS)
            add_f1_prior('PUHRF', 'SCAR', ratio, grp, SRF_F1_PRIOR_COLS)
    # SAR
    for d in data:
        path = f'./pu_trees_results/output_hellinger/results_{d}_sar.csv'
        df = pd.read_csv(path)
        for r, grp in df.groupby(["flip_ratio"]):
            ratio = r[0]
            # PU-HDT
            results['PUHDT']['SAR'][ratio]['roc_auc'].append(grp['roc_dt_puhd'].mean())
            results['PUHDT']['SAR'][ratio]['ap'].append(grp['ap_dt_puhd'].mean())
            results['PUHDT']['SAR'][ratio]['f1'].append(grp['f1_max_dt_puhd'].mean())
            results['PUHDT']['SAR'][ratio]['std_roc'].append(grp['roc_dt_puhd'].std())
            results['PUHDT']['SAR'][ratio]['std_ap'].append(grp['ap_dt_puhd'].std())
            results['PUHDT']['SAR'][ratio]['std_f1'].append(grp['f1_max_dt_puhd'].std())
            add_f1_05('PUHDT', 'SAR', ratio, grp, DT_F1_05_COLS)
            add_f1_prior('PUHDT', 'SAR', ratio, grp, DT_F1_PRIOR_COLS)
            # PU-HRF
            results['PUHRF']['SAR'][ratio]['roc_auc'].append(grp['roc_srf_puhd'].mean())
            results['PUHRF']['SAR'][ratio]['ap'].append(grp['ap_srf_puhd'].mean())
            results['PUHRF']['SAR'][ratio]['f1'].append(grp['f1_max_srf_puhd'].mean())
            results['PUHRF']['SAR'][ratio]['std_roc'].append(grp['roc_srf_puhd'].std())
            results['PUHRF']['SAR'][ratio]['std_ap'].append(grp['ap_srf_puhd'].std())
            results['PUHRF']['SAR'][ratio]['std_f1'].append(grp['f1_max_srf_puhd'].std())
            add_f1_05('PUHRF', 'SAR', ratio, grp, SRF_F1_05_COLS)
            add_f1_prior('PUHRF', 'SAR', ratio, grp, SRF_F1_PRIOR_COLS)


def gather_imbNNPU_results():
    data = ['cardio', 'climate', 'cover', 'letter', 'mammography', 'piechart', 'pizzacutter', 'poker',
            'satellite', 'segment', 'seismic', 'shuttle', 'thyroid', 'yeast']
    # SCAR
    for d in data:
        path = f'./pu_trees_results/output_imbnnpu/results_{d}_imbnnpu.csv'
        df = pd.read_csv(path)
        for r, grp in df.groupby(["flip_ratio"]):
            ratio = r[0]
            results['imbNNPU']['SCAR'][ratio]['roc_auc'].append(grp['roc_dt_puhd'].mean())
            results['imbNNPU']['SCAR'][ratio]['ap'].append(grp['ap_dt_puhd'].mean())
            results['imbNNPU']['SCAR'][ratio]['f1'].append(grp['f1_max_dt_puhd'].mean())
            results['imbNNPU']['SCAR'][ratio]['std_roc'].append(grp['roc_dt_puhd'].std())
            results['imbNNPU']['SCAR'][ratio]['std_ap'].append(grp['ap_dt_puhd'].std())
            results['imbNNPU']['SCAR'][ratio]['std_f1'].append(grp['f1_max_dt_puhd'].std())
            add_f1_05('imbNNPU', 'SCAR', ratio, grp, DT_F1_05_COLS)
            add_f1_prior('imbNNPU', 'SCAR', ratio, grp, DT_F1_PRIOR_COLS)
    # SAR
    for d in data:
        path = f'./pu_trees_results/output_imbnnpu/results_{d}_sar_imbnnpu.csv'
        df = pd.read_csv(path)
        for r, grp in df.groupby(["flip_ratio"]):
            ratio = r[0]
            results['imbNNPU']['SAR'][ratio]['roc_auc'].append(grp['roc_dt_puhd'].mean())
            results['imbNNPU']['SAR'][ratio]['ap'].append(grp['ap_dt_puhd'].mean())
            results['imbNNPU']['SAR'][ratio]['f1'].append(grp['f1_max_dt_puhd'].mean())
            results['imbNNPU']['SAR'][ratio]['std_roc'].append(grp['roc_dt_puhd'].std())
            results['imbNNPU']['SAR'][ratio]['std_ap'].append(grp['ap_dt_puhd'].std())
            results['imbNNPU']['SAR'][ratio]['std_f1'].append(grp['f1_max_dt_puhd'].std())
            add_f1_05('imbNNPU', 'SAR', ratio, grp, DT_F1_05_COLS)
            add_f1_prior('imbNNPU', 'SAR', ratio, grp, DT_F1_PRIOR_COLS)


def gather_nnpu_results():
    data = ['cardio', 'climate', 'cover', 'letter', 'mammography', 'piechart', 'pizzacutter', 'poker',
            'satellite', 'segment', 'seismic', 'shuttle', 'thyroid', 'yeast']
    # SCAR
    for d in data:
        path = f'./pu_trees_results/output_nnpu/results_{d}_nnpu.csv'
        df = pd.read_csv(path)
        for r, grp in df.groupby(["flip_ratio"]):
            ratio = r[0]
            results['nnPU']['SCAR'][ratio]['roc_auc'].append(grp['roc_dt_puhd'].mean())
            results['nnPU']['SCAR'][ratio]['ap'].append(grp['ap_dt_puhd'].mean())
            results['nnPU']['SCAR'][ratio]['f1'].append(grp['f1_max_dt_puhd'].mean())
            results['nnPU']['SCAR'][ratio]['std_roc'].append(grp['roc_dt_puhd'].std())
            results['nnPU']['SCAR'][ratio]['std_ap'].append(grp['ap_dt_puhd'].std())
            results['nnPU']['SCAR'][ratio]['std_f1'].append(grp['f1_max_dt_puhd'].std())
            add_f1_05('nnPU', 'SCAR', ratio, grp, DT_F1_05_COLS)
            add_f1_prior('nnPU', 'SCAR', ratio, grp, DT_F1_PRIOR_COLS)
    # SAR
    for d in data:
        path = f'./pu_trees_results/output_nnpu/results_{d}_sar_nnpu.csv'
        df = pd.read_csv(path)
        for r, grp in df.groupby(["flip_ratio"]):
            ratio = r[0]
            results['nnPU']['SAR'][ratio]['roc_auc'].append(grp['roc_dt_puhd'].mean())
            results['nnPU']['SAR'][ratio]['ap'].append(grp['ap_dt_puhd'].mean())
            results['nnPU']['SAR'][ratio]['f1'].append(grp['f1_max_dt_puhd'].mean())
            results['nnPU']['SAR'][ratio]['std_roc'].append(grp['roc_dt_puhd'].std())
            results['nnPU']['SAR'][ratio]['std_ap'].append(grp['ap_dt_puhd'].std())
            results['nnPU']['SAR'][ratio]['std_f1'].append(grp['f1_max_dt_puhd'].std())
            add_f1_05('nnPU', 'SAR', ratio, grp, DT_F1_05_COLS)
            add_f1_prior('nnPU', 'SAR', ratio, grp, DT_F1_PRIOR_COLS)


def gather_upu_results():
    data = ['cardio', 'climate', 'cover', 'letter', 'mammography', 'piechart', 'pizzacutter', 'poker',
            'satellite', 'segment', 'seismic', 'shuttle', 'thyroid', 'yeast']
    # SCAR
    for d in data:
        path = f'./pu_trees_results/output_upu/results_{d}_upu.csv'
        df = pd.read_csv(path)
        for r, grp in df.groupby(["flip_ratio"]):
            ratio = r[0]
            results['uPU']['SCAR'][ratio]['roc_auc'].append(grp['roc_dt_puhd'].mean())
            results['uPU']['SCAR'][ratio]['ap'].append(grp['ap_dt_puhd'].mean())
            results['uPU']['SCAR'][ratio]['f1'].append(grp['f1_max_dt_puhd'].mean())
            results['uPU']['SCAR'][ratio]['std_roc'].append(grp['roc_dt_puhd'].std())
            results['uPU']['SCAR'][ratio]['std_ap'].append(grp['ap_dt_puhd'].std())
            results['uPU']['SCAR'][ratio]['std_f1'].append(grp['f1_max_dt_puhd'].std())
            add_f1_05('uPU', 'SCAR', ratio, grp, DT_F1_05_COLS)
            add_f1_prior('uPU', 'SCAR', ratio, grp, DT_F1_PRIOR_COLS)
    # SAR
    for d in data:
        path = f'./pu_trees_results/output_upu/results_{d}_sar_upu.csv'
        df = pd.read_csv(path)
        for r, grp in df.groupby(["flip_ratio"]):
            ratio = r[0]
            results['uPU']['SAR'][ratio]['roc_auc'].append(grp['roc_dt_puhd'].mean())
            results['uPU']['SAR'][ratio]['ap'].append(grp['ap_dt_puhd'].mean())
            results['uPU']['SAR'][ratio]['f1'].append(grp['f1_max_dt_puhd'].mean())
            results['uPU']['SAR'][ratio]['std_roc'].append(grp['roc_dt_puhd'].std())
            results['uPU']['SAR'][ratio]['std_ap'].append(grp['ap_dt_puhd'].std())
            results['uPU']['SAR'][ratio]['std_f1'].append(grp['f1_max_dt_puhd'].std())
            add_f1_05('uPU', 'SAR', ratio, grp, DT_F1_05_COLS)
            add_f1_prior('uPU', 'SAR', ratio, grp, DT_F1_PRIOR_COLS)


def friedman_tests():
    pd.set_option("display.max_columns", None)  # never truncate columns
    pd.set_option("display.width", 250)  # or None to disable line wrapping
    pd.set_option("display.float_format", "{:.4f}".format)  # stop the mixed 1e-04 notation
    alpha = 0.05
    print('')
    print('Friedman tests [AVERAGE PRECISION]:')
    for labeling in ['SCAR', 'SAR']:
        for ratio in [0.25, 0.50, 0.75]:
            print(f'TESTING {labeling} {ratio}')
            statistic, p_value = friedmanchisquare(results['dif_iFPNU'][labeling][ratio]['ap'],
                                                   #results['eif_iFPNU'][labeling][ratio]['ap'],
                                                   #results['optiforest_iFPNU'][labeling][ratio]['ap'],
                                                   results['iFPU'][labeling][ratio]['ap'],
                                                   results['LBE'][labeling][ratio]['ap'],
                                                   #results['FL_lgb'][labeling][ratio]['ap'],
                                                   results['SAREM'][labeling][ratio]['ap'],
                                                   results['PUe'][labeling][ratio]['ap'],
                                                   results['uPU'][labeling][ratio]['ap'],
                                                   results['nnPU'][labeling][ratio]['ap'],
                                                   results['imbNNPU'][labeling][ratio]['ap'],
                                                   results['PUHDT'][labeling][ratio]['ap'],
                                                   results['PUHRF'][labeling][ratio]['ap'],
                                                   #results['DIF'][labeling][ratio]['ap'],
                                                   results['ALMPU'][labeling][ratio]['ap'],
                                                   )
            print(f'Statistic {statistic}, p-value {p_value}')
            print('')
            print('POST-HOC TEST:')
            # Combine three groups into one array
            data = np.array([results['dif_iFPNU'][labeling][ratio]['ap'],
                             #results['eif_iFPNU'][labeling][ratio]['ap'],
                             #results['optiforest_iFPNU'][labeling][ratio]['ap'],
                            results['iFPU'][labeling][ratio]['ap'],
                             results['LBE'][labeling][ratio]['ap'],
                             #results['FL_lgb'][labeling][ratio]['ap'],
                             results['SAREM'][labeling][ratio]['ap'],
                             results['PUe'][labeling][ratio]['ap'],
                             results['uPU'][labeling][ratio]['ap'],
                             results['nnPU'][labeling][ratio]['ap'],
                             results['imbNNPU'][labeling][ratio]['ap'],
                             results['PUHDT'][labeling][ratio]['ap'],
                             results['PUHRF'][labeling][ratio]['ap'],
                             #results['DIF'][labeling][ratio]['ap'],
                             results['ALMPU'][labeling][ratio]['ap']])
            # Conduct the Nemenyi post-hoc test
            print(sp.posthoc_nemenyi_friedman(data.T))
            print('')

    print('')
    print('Friedman tests [F1]:')
    for labeling in ['SCAR', 'SAR']:
        for ratio in [0.25, 0.50, 0.75]:
            print(f'TESTING {labeling} {ratio}')
            statistic, p_value = friedmanchisquare(results['dif_iFPNU'][labeling][ratio]['f1'],
                                                   #results['eif_iFPNU'][labeling][ratio]['f1'],
                                                   #results['optiforest_iFPNU'][labeling][ratio]['f1'],
                                                   results['iFPU'][labeling][ratio]['f1'],
                                                   results['LBE'][labeling][ratio]['f1'],
                                                   #results['FL_lgb'][labeling][ratio]['f1'],
                                                   results['SAREM'][labeling][ratio]['f1'],
                                                   results['PUe'][labeling][ratio]['f1'],
                                                   results['uPU'][labeling][ratio]['f1'],
                                                   results['nnPU'][labeling][ratio]['f1'],
                                                   results['imbNNPU'][labeling][ratio]['f1'],
                                                   results['PUHDT'][labeling][ratio]['f1'],
                                                   results['PUHRF'][labeling][ratio]['f1'],
                                                   #results['DIF'][labeling][ratio]['f1'],
                                                   results['ALMPU'][labeling][ratio]['f1'],
                                                   )
            print(f'Statistic {statistic}, p-value {p_value}')
            print('')
            print('POST-HOC TEST:')
            # Combine three groups into one array
            data = np.array([results['dif_iFPNU'][labeling][ratio]['f1'],
                             #results['eif_iFPNU'][labeling][ratio]['f1'],
                             #results['optiforest_iFPNU'][labeling][ratio]['f1'],
                             results['iFPU'][labeling][ratio]['f1'],
                             results['LBE'][labeling][ratio]['f1'],
                             #results['FL_lgb'][labeling][ratio]['f1'],
                             results['SAREM'][labeling][ratio]['f1'],
                             results['PUe'][labeling][ratio]['f1'],
                             results['uPU'][labeling][ratio]['f1'],
                             results['nnPU'][labeling][ratio]['f1'],
                             results['imbNNPU'][labeling][ratio]['f1'],
                             results['PUHDT'][labeling][ratio]['f1'],
                             results['PUHRF'][labeling][ratio]['f1'],
                             #results['DIF'][labeling][ratio]['f1'],
                             results['ALMPU'][labeling][ratio]['f1']])
            # Conduct the Nemenyi post-hoc test
            print(sp.posthoc_nemenyi_friedman(data.T))
            print('')


def produce_tables():
    METRICS = {"roc_auc": "std_roc", "ap": "std_ap", "f1": "std_f1"}
    METRICS.update({m: f'std_{m}' for m in F1_05_METRICS})
    METRICS.update({m: f'std_{m}' for m in F1_PRIOR_METRICS})
    METHOD_ORDER =["dif_iFPNU", "eif_iFPNU", "optiforest_iFPNU", "iFPU", "FL_lgb", "LBE", "SAREM", "PUe",
                    "uPU", "nnPU", "imbNNPU", "PUHDT", "PUHRF", "DIF", "ALMPU"]
    MECH_LABEL = {"SAR": "knn (SAR)", "SCAR": "uniform (SCAR)"}

    def pooled_std(s):
        """RMS of per-dataset stds, ignoring NaN (= pooled within-dataset std)."""
        if s.isna().all():
            return np.nan  # e.g. a method without the F1@prior metrics yet
        return np.sqrt(np.nanmean(np.square(s)))

    rows = []
    for method, by_mech in results.items():
        for mech, by_ratio in by_mech.items():
            for ratio, d in by_ratio.items():
                for metric, std_key in METRICS.items():
                    for i, (m, s) in enumerate(zip(d[metric], d[std_key])):
                        rows.append({"method": method, "label_noise": MECH_LABEL[mech],
                                     "flip_ratio": ratio, "metric": metric, "dataset": i,
                                     "mean": m, "std": s})
    df = pd.DataFrame(rows)

    keys = ["label_noise", "flip_ratio", "method"]
    pd.set_option("display.width", 250)

    tables = {}
    for metric in METRICS:
        sub = df[df["metric"] == metric]
        agg = sub.groupby(keys).agg(mean=("mean", "mean"), std=("std", pooled_std))
        # Only the methods with results for this metric (e.g. not all methods have the F1@0.5 metrics).
        # A method missing the metric for some dataset/setting is left out, never averaged over fewer datasets.
        incomplete = set(sub.loc[sub["mean"].isna(), "method"])
        methods = [m for m in METHOD_ORDER if m in set(sub["method"]) and m not in incomplete]
        tbl = (agg.unstack("method").swaplevel(axis=1)
               .reindex(columns=pd.MultiIndex.from_product([methods, ["mean", "std"]])))
        tables[metric] = tbl
        print(f"\n{metric} -- mean over datasets | std = std over partitions, pooled over datasets")
        print("-" * 78)
        print(tbl.round(4).to_string())
        if incomplete:
            print(f"(left out, {metric} missing for some datasets/settings: "
                  f"{', '.join(m for m in METHOD_ORDER if m in incomplete)})")
        # tbl.round(3).to_latex() for the manuscript


if __name__ == '__main__':
    #parse_result_csv('./output/SCALED_CARDIO_SCAR_0.25_iFPU_lgb/results.csv')
    gather_results()
    gather_hellinger_trees_results()
    gather_imbNNPU_results()
    gather_nnpu_results()
    gather_upu_results()
    #print(results)
    produce_tables()
    friedman_tests()
