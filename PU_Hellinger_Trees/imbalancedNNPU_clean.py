from pos_noisyneg.spyEM import SpyEM
from pos_noisyneg.elkannoto import WeightedElkanotoPuClassifier
from pos_noisyneg.rankpruning import RankPruning
from pos_noisyneg.utils import make_noisy_negatives
from pos_noisyneg.PU_bagging import BaggingPuClassifier
from pu_tree_simplified._pu_randomforest import PURandomForestClassifier as PURF_SIMP
from pu_tree_simplified._pu_classes import DecisionTreeClassifier

from ecsmodels.methodologies.cs_boost import CSBoost

from experiment_datasets import (FLIP_RATIOS, SKIPPABLE, load_dataset, output_path,
                                 report_sweep, sweep_units, write_unit)

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, precision_recall_curve
from sklearn.model_selection import ParameterGrid
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from imblearn.over_sampling import ADASYN

import os

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.special import expit

OUT_DIR = 'output_imbnnpu'
MODEL_SUFFIX = '_imbnnpu'
SKIP_EXISTING = False

# Metrics land in `*_dt_puhd` columns copied from the PUHD script; the output
# directory, not the column name, is what identifies this model.
COLUMNS = ['data_partition', 'dataset', 'flip_ratio', 'label_noise', 'roc_dt_puhd',
           'ap_dt_puhd', 'f1_dt_puhd', 'f1_ci_dt_puhd', 'f1_max_dt_puhd', 'f1_pos_dt_puhd',
           'f1_neg_dt_puhd', 'f1_macro_dt_puhd', 'f1_ci_pos_dt_puhd', 'f1_ci_neg_dt_puhd',
           'f1_ci_macro_dt_puhd']

set_thresholds = np.linspace(0.01, 0.99, num=100)


def binarize_prob(prob, threshold=0.50):
    labels = np.zeros(prob.shape, dtype=int)
    labels[prob > threshold] = 1
    return labels


list_random_state = list(np.arange(10))

os.makedirs(OUT_DIR, exist_ok=True)
written, skipped_existing, failures = [], [], []

for filename, stem, label_noise, noise_suffix in tqdm(sweep_units(), desc='dataset x noise'):

    path = output_path(OUT_DIR, stem, noise_suffix, MODEL_SUFFIX)
    if SKIP_EXISTING and os.path.exists(path):
        skipped_existing.append(path)
        continue

    raw = load_dataset(filename)
    rows = []

    for r in list_random_state:

        d = raw.copy()
        np.random.shuffle(d)

        #if len(d) > 10000:
        #    rs = StratifiedShuffleSplit(n_splits=1, train_size=10000, random_state=r)
        #    aux_tr, aux_ts = [(a, b) for a, b in rs.split(d[:, :-1], d[:, -1])][0]
        #    d = d[aux_tr]

        X = d[:, :-1]
        y = d[:, -1]

        for flip_ratio in FLIP_RATIOS:

            row = {'data_partition': r, 'dataset': stem,
                   'flip_ratio': flip_ratio, 'label_noise': label_noise}

            try:
                noisy_y = make_noisy_negatives(y,
                                               X=X,
                                               flip_ratio=flip_ratio,
                                               label_noise=label_noise,
                                               random_state=r)
                ## Class Y Generation
                class_y = y.copy()
                class_y[np.logical_and(np.array(noisy_y == 0), np.array(y == 1))] = 2

                conditions = [class_y == 0, class_y == 2, class_y == 1]
                values = ['negs', 'noisy_negs', 'pos']
                class_label = np.select(conditions, values)
                contamination_ratio = np.mean(class_label == 'noisy_negs')

                rs = StratifiedShuffleSplit(n_splits=1, test_size=0.30,
                                            random_state=r)
                ix_tr, ix_ts = [(a, b) for a, b in rs.split(X, class_label)][0]

                rs_trval = StratifiedShuffleSplit(n_splits=1, test_size=0.50,
                                                  random_state=r)
                ix_tr_tr, ix_tr_val = [(a, b) for a, b in rs_trval.split(X[ix_tr], class_label[ix_tr])][0]

                prior_y = np.mean(y)

                ### MODELING ###
                prior_s = np.mean(noisy_y)
                c = prior_s / prior_y

                cost_matrix_tr_tr = np.zeros((len(X[ix_tr_tr]), 2, 2))  # cost_matrix [[TN, FN], [FP, TP]]
                cost_matrix_tr_tr[:, 0, 0] = 0.0
                cost_matrix_tr_tr[:, 0, 1] = 1.0
                cost_matrix_tr_tr[:, 1, 0] = 1.0
                cost_matrix_tr_tr[:, 1, 1] = 0.0

                m1_xgb_imbnnpu = CSBoost(obj='imbnnpu',
                                         oversampling=0.50,
                                         prior_y=np.mean(y),
                                         prob_lab=np.mean(noisy_y),
                                         validation=False,
                                         random_state=r).fit(X[ix_tr_tr],
                                                             noisy_y[ix_tr_tr],
                                                             cost_matrix_train=cost_matrix_tr_tr)

                prob = expit(m1_xgb_imbnnpu.inplace_predict(X[ix_tr_val]))

                set_pred_labels = [binarize_prob(prob, threshold=i) for i in set_thresholds]
                set_f1_scores = [f1_score(y[ix_tr_val], i) for i in set_pred_labels]
                max_f1score = np.max(set_f1_scores)
                ix_fscore = np.where(set_f1_scores == max_f1score)[0][0]
                opt_threshold = set_thresholds[ix_fscore]

                # Training optimal
                # Cost Matrix for Training
                cost_matrix_tr = np.zeros((len(X[ix_tr]), 2, 2))  # cost_matrix [[TN, FN], [FP, TP]]
                cost_matrix_tr[:, 0, 0] = 0.0
                cost_matrix_tr[:, 0, 1] = 1.0
                cost_matrix_tr[:, 1, 0] = 1.0
                cost_matrix_tr[:, 1, 1] = 0.0

                m1_xgb_imbnnpu = CSBoost(obj='imbnnpu',
                                         oversampling=0.50,
                                         prior_y=np.mean(y),
                                         prob_lab=np.mean(noisy_y),
                                         validation=False,
                                         random_state=r).fit(X[ix_tr],
                                                             noisy_y[ix_tr],
                                                             cost_matrix_train=cost_matrix_tr)

                m1_imbnnpu_prob = expit(m1_xgb_imbnnpu.inplace_predict(X[ix_ts]))
                m1_imbnnpu_label = np.array([m1_imbnnpu_prob[i] >= 0.50 for i in range(m1_imbnnpu_prob.shape[0])])
                label_ci_imbnnpu = np.array([m1_imbnnpu_prob[i] >= prior_y for i in range(m1_imbnnpu_prob.shape[0])])
                label_f1max_imbnnpu = np.array(
                    [m1_imbnnpu_prob[i] >= opt_threshold for i in range(m1_imbnnpu_prob.shape[0])])

                row['roc_dt_puhd'] = roc_auc_score(y[ix_ts], m1_imbnnpu_prob)
                row['ap_dt_puhd'] = average_precision_score(y[ix_ts], m1_imbnnpu_prob)
                row['f1_dt_puhd'] = f1_score(y[ix_ts], m1_imbnnpu_label)
                row['f1_ci_dt_puhd'] = f1_score(y[ix_ts], label_ci_imbnnpu)
                row['f1_max_dt_puhd'] = f1_score(y[ix_ts], label_f1max_imbnnpu)
                row['f1_pos_dt_puhd'] = f1_score(y[ix_ts], m1_imbnnpu_label, pos_label=1)
                row['f1_neg_dt_puhd'] = f1_score(y[ix_ts], m1_imbnnpu_label, pos_label=0)
                row['f1_macro_dt_puhd'] = f1_score(y[ix_ts], m1_imbnnpu_label, average='macro')
                row['f1_ci_pos_dt_puhd'] = f1_score(y[ix_ts], label_ci_imbnnpu, pos_label=1)
                row['f1_ci_neg_dt_puhd'] = f1_score(y[ix_ts], label_ci_imbnnpu, pos_label=0)
                row['f1_ci_macro_dt_puhd'] = f1_score(y[ix_ts], label_ci_imbnnpu, average='macro')

                print('PR AUC imbnnPU: ', row['ap_dt_puhd'])

                rows.append(row)

            except SKIPPABLE as exc:
                failures.append((stem, label_noise, int(r), flip_ratio,
                                 f'{type(exc).__name__}: {exc}'))
                continue

    if not rows:
        print(f'[skip] {stem}/{label_noise}: every iteration failed, no file written',
              flush=True)
        continue

    write_unit(path, rows, COLUMNS)
    written.append((path, len(rows)))
    print(f'[ok] {path}  {len(rows)} rows', flush=True)

report_sweep(written, skipped_existing, failures,
             len(list_random_state) * len(FLIP_RATIOS))
