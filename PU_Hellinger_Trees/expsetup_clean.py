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

OUT_DIR = 'output'
MODEL_SUFFIX = ''
SKIP_EXISTING = False

# `f1_max_test_*` is the oracle threshold tuned on test -- an upper bound, not a
# fair score. Kept so output/ carries every metric; readers ignore what they don't use.
COLUMNS = ['data_partition', 'dataset', 'flip_ratio', 'label_noise', 'roc_dt_puhd',
           'ap_dt_puhd', 'f1_dt_puhd', 'f1_ci_dt_puhd', 'f1_max_dt_puhd',
           'f1_max_test_dt_puhd', 'f1_pos_dt_puhd', 'f1_neg_dt_puhd', 'f1_macro_dt_puhd',
           'f1_ci_pos_dt_puhd', 'f1_ci_neg_dt_puhd', 'f1_ci_macro_dt_puhd', 'roc_srf_puhd',
           'ap_srf_puhd', 'f1_srf_puhd', 'f1_ci_srf_puhd', 'f1_max_srf_puhd',
           'f1_max_test_srf_puhd', 'f1_pos_srf_puhd', 'f1_neg_srf_puhd', 'f1_macro_srf_puhd',
           'f1_ci_pos_srf_puhd', 'f1_ci_neg_srf_puhd', 'f1_ci_macro_srf_puhd']

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

            # Both models share one row and one try: a failure between them must
            # discard the whole row, never leave the two metric groups uneven.
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

                ### MODELING ####
                ## Elkan-Noto PU Hellinger Distance Decision Tree ##
                # F1 Score Maximization
                m_dt_puhd = DecisionTreeClassifier(random_state=r, max_depth=5)
                m_dt_puhd.fit(X[ix_tr_tr], noisy_y[ix_tr_tr], p_y=np.mean(noisy_y))
                prob = m_dt_puhd.predict_proba(X[ix_tr_val])[:, 1]

                set_pred_labels = [binarize_prob(prob, threshold=i) for i in set_thresholds]
                set_f1_scores = [f1_score(y[ix_tr_val], i) for i in set_pred_labels]
                max_f1score = np.max(set_f1_scores)
                ix_fscore = np.where(set_f1_scores == max_f1score)[0][0]
                opt_threshold = set_thresholds[ix_fscore]

                # Training optimal
                m_dt_puhd = DecisionTreeClassifier(random_state=r, max_depth=5)
                m_dt_puhd.fit(X[ix_tr], noisy_y[ix_tr], p_y=np.mean(y))

                prob_dt_puhd = m_dt_puhd.predict_proba(X[ix_ts])[:, 1]
                label_dt_puhd = binarize_prob(prob_dt_puhd)
                label_ci_dt_puhd = binarize_prob(prob_dt_puhd, threshold=prior_y)
                label_f1max_dt_puhd = binarize_prob(prob_dt_puhd, threshold=opt_threshold)

                # Find the threshold that gives the best F1-score in test
                p, rec, thresholds = precision_recall_curve(y[ix_ts], prob_dt_puhd)
                numerator = 2 * rec * p
                denom = rec + p
                f1_scores = np.divide(numerator, denom, out=np.zeros_like(denom), where=(denom != 0))
                best_f1 = np.max(f1_scores)

                row['roc_dt_puhd'] = roc_auc_score(y[ix_ts], prob_dt_puhd)
                row['ap_dt_puhd'] = average_precision_score(y[ix_ts], prob_dt_puhd)
                row['f1_dt_puhd'] = f1_score(y[ix_ts], label_dt_puhd)
                row['f1_ci_dt_puhd'] = f1_score(y[ix_ts], label_ci_dt_puhd)
                row['f1_max_dt_puhd'] = f1_score(y[ix_ts], label_f1max_dt_puhd)
                row['f1_max_test_dt_puhd'] = best_f1
                row['f1_pos_dt_puhd'] = f1_score(y[ix_ts], label_dt_puhd, pos_label=1)
                row['f1_neg_dt_puhd'] = f1_score(y[ix_ts], label_dt_puhd, pos_label=0)
                row['f1_macro_dt_puhd'] = f1_score(y[ix_ts], label_dt_puhd, average='macro')
                row['f1_ci_pos_dt_puhd'] = f1_score(y[ix_ts], label_ci_dt_puhd, pos_label=1)
                row['f1_ci_neg_dt_puhd'] = f1_score(y[ix_ts], label_ci_dt_puhd, pos_label=0)
                row['f1_ci_macro_dt_puhd'] = f1_score(y[ix_ts], label_ci_dt_puhd, average='macro')
                print('Av.Precision PU HDT score: ', row['ap_dt_puhd'])

                ## Elkan-Noto Stratified Bootstrap Random Forest Hellinger Distance ##
                # F1 Score Maximization
                m_srf_puhd = PURF_SIMP(random_state=r, max_samples=sum(noisy_y[ix_tr_tr] == 0),
                                       pu_biased_bootstrap=True)
                m_srf_puhd.fit(X[ix_tr_tr], noisy_y[ix_tr_tr], p_y=np.mean(y))

                prob = m_srf_puhd.predict_proba(X[ix_tr_val])[:, 1]

                set_pred_labels = [binarize_prob(prob, threshold=i) for i in set_thresholds]
                set_f1_scores = [f1_score(y[ix_tr_val], i) for i in set_pred_labels]
                max_f1score = np.max(set_f1_scores)
                ix_fscore = np.where(set_f1_scores == max_f1score)[0][0]
                opt_threshold = set_thresholds[ix_fscore]

                # Training optimal
                m_srf_puhd = PURF_SIMP(random_state=r, max_samples=sum(noisy_y[ix_tr] == 0),
                                       pu_biased_bootstrap=True)
                m_srf_puhd.fit(X[ix_tr], noisy_y[ix_tr], p_y=np.mean(y))

                prob_srf_puhd = m_srf_puhd.predict_proba(X[ix_ts])[:, 1]
                label_srf_puhd = binarize_prob(prob_srf_puhd)
                label_ci_srf_puhd = binarize_prob(prob_srf_puhd, threshold=prior_y)
                label_f1max_srf_puhd = binarize_prob(prob_srf_puhd, threshold=opt_threshold)

                # Find the threshold that gives the best F1-score in test
                p, rec, thresholds = precision_recall_curve(y[ix_ts], prob_srf_puhd)
                numerator = 2 * rec * p
                denom = rec + p
                f1_scores = np.divide(numerator, denom, out=np.zeros_like(denom), where=(denom != 0))
                best_f1 = np.max(f1_scores)

                row['roc_srf_puhd'] = roc_auc_score(y[ix_ts], prob_srf_puhd)
                row['ap_srf_puhd'] = average_precision_score(y[ix_ts], prob_srf_puhd)
                row['f1_srf_puhd'] = f1_score(y[ix_ts], label_srf_puhd)
                row['f1_ci_srf_puhd'] = f1_score(y[ix_ts], label_ci_srf_puhd)
                row['f1_max_srf_puhd'] = f1_score(y[ix_ts], label_f1max_srf_puhd)
                row['f1_max_test_srf_puhd'] = best_f1
                row['f1_pos_srf_puhd'] = f1_score(y[ix_ts], label_srf_puhd, pos_label=1)
                row['f1_neg_srf_puhd'] = f1_score(y[ix_ts], label_srf_puhd, pos_label=0)
                row['f1_macro_srf_puhd'] = f1_score(y[ix_ts], label_srf_puhd, average='macro')
                row['f1_ci_pos_srf_puhd'] = f1_score(y[ix_ts], label_ci_srf_puhd, pos_label=1)
                row['f1_ci_neg_srf_puhd'] = f1_score(y[ix_ts], label_ci_srf_puhd, pos_label=0)
                row['f1_ci_macro_srf_puhd'] = f1_score(y[ix_ts], label_ci_srf_puhd, average='macro')
                print('Av.Precision Stratified PURF score: ', row['ap_srf_puhd'])

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
