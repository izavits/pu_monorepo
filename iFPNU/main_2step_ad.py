#
# main_2step_ad.py
# 2-step PU learning based on anomaly detection as first step
#

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_curve, auc
from models.abif import AttentionBasedIsolationForest
from models.basic_abif_model import Mode
from data_loaders.csv_loader import CSVDataset
from data_loaders.array_loader import ArrayDataset
from torch.utils.data import DataLoader
from data_loaders.pu_data import create_pu_data
from models.gradient_boost import PytorchBasedGenericGradientBoost
from train_and_validation import train, validate, validate_xgb, f1_at_prior_threshold
from models.mlp import MLP
from pu.pnu_losses import SigmoidNPU, FocalLossNPU
import eif as iso
import numpy as np
import logging
import configparser
import torch
import sys
from pyod.models.dif import DIF
from pyod.models.lof import LOF
from models.optiforest import OptIForest
from pyod.models.iforest import IForest
import xgboost as xgb
from pu.fpu_xgb import nnpnu_focal_loss, pnu_focal_loss
import lightgbm as lgb
import pandas as pd
import warnings

warnings.filterwarnings('ignore')  # Suppress all warnings


def prepare_pu_data_plain(path, pu_ratio, labeling_mechanism, random_state):
    """
    Prepare data: load the dataset, create splits, create PU data in training set
    and return the corresponding data
    :param path: path to dataset
    :param pu_ratio: the ratio of the positive labeled examples
    :param labeling_mechanism: the labeling mechanism assumption
    :param random_state: random state for splitting
    :return: train and test data
    """
    # Load the dataset and get split
    dataset = CSVDataset(path)
    train, test = dataset.get_splits(random_state=random_state)
    # Create PU data
    x_train = train.dataset.X[train.indices]
    y_train = train.dataset.y[train.indices]
    loss_prior = np.mean(y_train)
    x_test = test.dataset.X[test.indices]
    y_test = test.dataset.y[test.indices]
    y_ = create_pu_data(x_train, y_train, pu_ratio, labeling_mechanism, None, random_state=random_state)
    y_ = y_.flatten()
    y_train = y_train.flatten()
    y_test = y_test.flatten()
    return x_train, y_, x_test, y_test, y_train, loss_prior


def log_metrics(y_test, y_pred):
    confusion = confusion_matrix(y_test, y_pred)
    TN, FP, FN, TP = confusion.flat
    TPR = TP / (TP + FN)
    FPR = FP / (FP + TN)
    normal = y_test[y_test == 0]
    anomalies = y_test[y_test == 1]
    F1 = f1_score(y_test, y_pred)
    logger.info(
        f'Proportion anomalies/normal = {len(anomalies)}/{len(normal)} = {(len(anomalies) / len(normal)) * 100:.1f}%')
    logger.info(f'F1 score {F1:.4f}')
    logger.info(f'TPR score {TPR:.4f}, FPR {FPR:.4f}')
    logger.info(f'TP {TP:.4f}, FP {FP:.4f}')
    logger.info(f'FN {FN:.4f}, TN {TN:.4f}')
    fpr, tpr, thresholds = roc_curve(y_test, y_pred, pos_label=1)
    logger.info(f'AUC {auc(fpr, tpr):.4f}')
    logger.info(f'ACCURACY {accuracy_score(y_pred, y_test):.4f}')


def pnu_statistics(y_train, y_pu, y_pnu):
    logger.info('---------------')
    logger.info(f'PNU Statistics')
    logger.info('---------------')
    pn_pos = np.count_nonzero(y_train == 1.0)
    pn_pos_percent = (pn_pos / len(y_train)) * 100
    pn_neg = np.count_nonzero(y_train == 0.0)
    logger.info(f'[PN stats]--> POS: {pn_pos}, POS_PERCENTAGE: {pn_pos_percent:.2f}%, NEG: {pn_neg}')
    pu_pos = np.count_nonzero(y_pu == 1.0)
    pu_pos_percent = (pu_pos / len(y_pu)) * 100
    pu_unlabeled = np.count_nonzero(y_pu == 0.0)
    logger.info(
        f'[PU stats]--> POS: {pu_pos}, POS_PERCENTAGE: {pu_pos_percent:.2f}%, UNLABELED: {pu_unlabeled}')
    pnu_pos = np.count_nonzero(y_pnu == 1.0)
    pnu_pos_percent = (pnu_pos / len(y_pnu)) * 100
    pnu_unlabeled = np.count_nonzero(y_pnu == -1.0)
    pnu_neg = np.count_nonzero(y_pnu == 0.0)
    logger.info(
        f'[PNU stats]--> POS: {pnu_pos}, POS_PERCENTAGE: {pnu_pos_percent:.2f}%, UNLABELED: {pnu_unlabeled}, NEG: {pnu_neg}')
    hidden_pos = pn_pos - pu_pos
    # Check how many of the hidden positives (unlabeled) were detected as unreliable negatives
    # and therefore remained unlabeled
    cnt_pnu_unlabeled = 0
    for i in range(len(y_pu)):
        if y_pu[i] == 0.0 and y_train[i] == 1.0:
            # if it was a hidden positive (i.e. unlabeled), check if remained unlabeled
            if y_pnu[i] == -1.0:
                cnt_pnu_unlabeled += 1
    pnu_pos_percent_unlabeled = (cnt_pnu_unlabeled / hidden_pos) * 100
    logger.info(
        f'[PNU]--> Hidden positives that remained unlabeled: {cnt_pnu_unlabeled}, i.e. {pnu_pos_percent_unlabeled:.2f}% of pos unlabeled')
    # Check how many of the hidden positives (unlabeled) were detected (falsely) as reliable negatives
    # and therefore labeled (falsely) as negative
    cnt_pnu_negative = 0
    for i in range(len(y_pu)):
        if y_pu[i] == 0.0 and y_train[i] == 1.0:
            # if it was a hidden positive (i.e. unlabeled), check if it is now detected as negative
            if y_pnu[i] == 0.0:
                cnt_pnu_negative += 1
    pnu_pos_percent_negative = (cnt_pnu_negative / hidden_pos) * 100
    logger.info(
        f'[PNU]--> Hidden positives that became negative: {cnt_pnu_negative}, i.e. {pnu_pos_percent_negative:.2f}% of pos unlabeled')


def pnu_data_metrics(y_orig, y_pnu):
    """
    Metrics for pnu data.
    :param y_orig: original labels
    :param y_pnu: predicted labels
    """
    # Calculate percentage of labeled examples
    num_instances = len(y_orig)
    num_negatives = 0
    for i in y_orig:
        if i == 0.0:
            num_negatives += 1
    cnt_labeled = 0
    for i in y_pnu:
        if i != -1.0:
            cnt_labeled += 1
    perc_labeled = (cnt_labeled / num_instances) * 100
    # Calculate the precision for labeled
    num_correct = 0
    for i in range(num_instances):
        if y_pnu[i] != -1.0:
            if y_pnu[i] == y_orig[i]:
                num_correct += 1
    precision_labeled = (num_correct / cnt_labeled) * 100
    # Calculate the recall of reliable negatives
    num_neg_correct = 0
    for i in range(num_instances):
        if y_pnu[i] == 0.0 and y_orig[i] == 0.0:
            num_neg_correct += 1
    recall_labeled = (num_neg_correct / num_negatives) * 100
    return perc_labeled, precision_labeled, recall_labeled


def average_results(accuracies, aucs, rprecisions, precisions, recalls, pr_areas, f1, f1_pos, f1_neg, f1_macro,
                    f1_pos_prior, f1_neg_prior, f1_macro_prior):
    """
    Average individual results
    :param accuracies: list of accuracy results
    :param aucs: list of auc results
    :param rprecisions: list of rprecision results
    :param precisions: list of precision results
    :param recalls: list of recall results
    :param pr_areas: list of pr_area results
    :param f1_pos: list of f1 results at the 0.5 threshold (pos)
    :param f1_neg: list of f1 results at the 0.5 threshold (neg)
    :param f1_macro: list of f1 results at the 0.5 threshold (macro)
    :param f1_pos_prior: list of f1 results at the class-prior threshold (pos)
    :param f1_neg_prior: list of f1 results at the class-prior threshold (neg)
    :param f1_macro_prior: list of f1 results at the class-prior threshold (macro)
    :return: average results for all metrics, plus standard deviation and standard error
    """
    avg_results = {
        'mean_acc': np.mean(accuracies),
        'stdev_acc': np.std(accuracies),
        'stder_acc': np.std(accuracies) / np.sqrt(np.size(accuracies)),
        'mean_auc': np.mean(aucs),
        'stdev_auc': np.std(aucs),
        'stder_auc': np.std(aucs) / np.sqrt(np.size(aucs)),
        'mean_rprecision': np.mean(rprecisions),
        'stdev_rprecision': np.std(rprecisions),
        'stder_rprecision': np.std(rprecisions) / np.sqrt(np.size(rprecisions)),
        'mean_precision': np.mean(precisions),
        'stdev_precision': np.std(precisions),
        'stder_precision': np.std(precisions) / np.sqrt(np.size(precisions)),
        'mean_recall': np.mean(recalls),
        'stdev_recall': np.std(recalls),
        'stder_recall': np.std(recalls) / np.sqrt(np.size(recalls)),
        'mean_pr_area': np.mean(pr_areas),
        'stdev_pr_area': np.std(pr_areas),
        'stder_pr_area': np.std(pr_areas) / np.sqrt(np.size(pr_areas)),
        'mean_f1': np.mean(f1),
        'stdev_f1': np.std(f1),
        'stder_f1': np.std(f1) / np.sqrt(np.size(f1)),
        'mean_f1_pos': np.mean(f1_pos),
        'stdev_f1_pos': np.std(f1_pos),
        'stder_f1_pos': np.std(f1_pos) / np.sqrt(np.size(f1_pos)),
        'mean_f1_neg': np.mean(f1_neg),
        'stdev_f1_neg': np.std(f1_neg),
        'stder_f1_neg': np.std(f1_neg) / np.sqrt(np.size(f1_neg)),
        'mean_f1_macro': np.mean(f1_macro),
        'stdev_f1_macro': np.std(f1_macro),
        'stder_f1_macro': np.std(f1_macro) / np.sqrt(np.size(f1_macro)),
        'mean_f1_pos_prior': np.mean(f1_pos_prior),
        'stdev_f1_pos_prior': np.std(f1_pos_prior),
        'stder_f1_pos_prior': np.std(f1_pos_prior) / np.sqrt(np.size(f1_pos_prior)),
        'mean_f1_neg_prior': np.mean(f1_neg_prior),
        'stdev_f1_neg_prior': np.std(f1_neg_prior),
        'stder_f1_neg_prior': np.std(f1_neg_prior) / np.sqrt(np.size(f1_neg_prior)),
        'mean_f1_macro_prior': np.mean(f1_macro_prior),
        'stdev_f1_macro_prior': np.std(f1_macro_prior),
        'stder_f1_macro_prior': np.std(f1_macro_prior) / np.sqrt(np.size(f1_macro_prior))
    }
    return avg_results


def average_pnu_data_metrics_results(percents, precisions, recalls):
    """
    Average individual results for pnu data.
    """
    avg_results = {
        'mean_percent': np.mean(percents),
        'stdev_percent': np.std(percents),
        'mean_precision': np.mean(precisions),
        'stdev_precision': np.std(precisions),
        'mean_recall': np.mean(recalls),
        'stdev_recall': np.std(recalls)
    }
    return avg_results


class ColorHandler(logging.StreamHandler):
    GRAY8 = "38;5;8"
    GRAY7 = "38;5;7"
    ORANGE = "33"
    RED = "31"
    WHITE = "0"

    def emit(self, record):
        # Don't use white for any logging, to help distinguish from user print statements
        level_color_map = {
            logging.DEBUG: self.GRAY8,
            logging.INFO: self.GRAY7,
            logging.WARNING: self.ORANGE,
            logging.ERROR: self.RED,
        }
        csi = f"{chr(27)}["  # control sequence introducer
        color = level_color_map.get(record.levelno, self.WHITE)
        print(f"{csi}{color}m{record.msg}{csi}m")


def my_custom_logger(output, output_log_name):
    """
    Method to return a custom logger with the given name and level
    """
    logFormatter = logging.Formatter(
        "%(asctime)s |%(name)s| [%(levelname)-5.5s]  %(message)s")
    logger = logging.getLogger('Main')
    fileHandler = logging.FileHandler(
        "{0}/{1}.log".format(output, output_log_name))
    fileHandler.setFormatter(logFormatter)
    logger.addHandler(fileHandler)
    consoleHandler = logging.StreamHandler(stream=sys.stdout)
    consoleHandler.setFormatter(logFormatter)
    logger.addHandler(ColorHandler())
    logger.setLevel(logging.INFO)
    return logger


if __name__ == "__main__":
    # Read configuration file
    config = configparser.ConfigParser()
    config.read('./config.ini')
    experiment_name = config['RUN_2STEP_AD']['EXPERIMENT']
    output = config['RUN_2STEP_AD']['OUTPUT']
    model_name = config['RUN_2STEP_AD']['MODEL']
    classifier_name = config['RUN_2STEP_AD']['CLASSIFIER']
    criterion_name = config['RUN_2STEP_AD']['CRITERION']
    device_type = config['RUN_2STEP_AD']['DEVICE']
    pu_ratio = float(config['RUN_2STEP_AD']['PU_RATIO'])
    loss_alpha = float(config['LOSS_PARAMS']['ALPHA'])
    loss_gamma = float(config['LOSS_PARAMS']['GAMMA'])
    loss_beta = float(config['LOSS_PARAMS']['BETA'])
    loss_eta = float(config['RUN_2STEP_AD']['ETA'])
    labeling_mechanism = config['RUN_2STEP_AD']['LABELING_MECHANISM']
    num_runs = int(config['RUN_2STEP_AD']['NUM_RUNS'])
    max_epochs = int(config['RUN_2STEP_AD']['MAX_EPOCHS'])
    patience = int(config['RUN_2STEP_AD']['PATIENCE'])
    log_interval = int(config['RUN_2STEP_AD']['LOG_INTERVAL'])
    prior = config['RUN_2STEP_AD']['PRIOR']
    save_model = config['RUN_2STEP_AD'].getboolean('SAVE_BEST_MODEL')
    run_all = config['RUN_2STEP_AD'].getboolean('RUN_ALL')
    device = torch.device(device_type)
    if prior == 'p':
        output_log_name = f'{experiment_name}_{pu_ratio}_{labeling_mechanism}_{num_runs}_' \
                          f'{criterion_name}_{model_name}'
    else:
        output_log_name = f'{experiment_name}_{pu_ratio}_{labeling_mechanism}_{num_runs}_' \
                          f'{criterion_name}_{model_name}_{prior}'

    EXPERIMENTS = []
    if run_all:
        for key in config['DATASETS'].keys():
            EXPERIMENTS.append(key.upper())
    else:
        EXPERIMENTS.append(experiment_name)

    all_experimental_results = {}
    all_experimental_PNU_data_results = {}
    for experiment_name in EXPERIMENTS:
        dataset_path = config['DATASETS'][experiment_name]
        output_log_name = f'{experiment_name}_{pu_ratio}_{labeling_mechanism}_{num_runs}_' \
                          f'{model_name}_2STEP_AD'
        # Set up logging
        logger = my_custom_logger(output, output_log_name)
        # Setup variables for holding results
        acc_all = []
        auc_all = []
        rprecision_all = []
        precision_all = []
        recall_all = []
        pr_area_all = []
        f1_all = []
        f1_pos_all = []
        f1_neg_all = []
        f1_macro_all = []
        f1_pos_prior_all = []
        f1_neg_prior_all = []
        f1_macro_prior_all = []

        pnu_percent_labeled_all = []
        pnu_precision_labeled_all = []
        pnu_recall_labeled_all = []

        logger.info(f'EXPERIMENT: {experiment_name}')
        logger.info(f'PU_RATIO: {pu_ratio} {labeling_mechanism} | Runs: {num_runs} | '
                    f'Model: {model_name}')
        logger.info(f'===========================================================')
        for r in range(num_runs):
            logger.info(f'>>>>>>>>>> Run {r + 1} <<<<<<<<<<')
            # Prepare PU data
            X_train, y_, X_test, y_test, y_train, loss_prior = prepare_pu_data_plain(dataset_path, pu_ratio,
                                                                                     labeling_mechanism,
                                                                                     random_state=r)
            # Experiment with different priors
            #loss_prior = 4.0*loss_prior
            logger.info(
                f'Training set length: {len(X_train)} | Test set length: {len(X_test)}')
            # Instantiate the model
            if model_name == 'abif':
                model = AttentionBasedIsolationForest(mode=1)
                model.setup()
                model.create_tree(seed=r, n_estimators=100, max_samples=len(X_train))
                # Fit model in PU data
                logger.info('[1st step] Fitting AD model in PU data..')
                model.fit(X_train, y_)
                if model.get_type_model() == Mode.ATTENTION:
                    logger.info('[1st step] Optimizing weights..')
                    model.optimize_weights(X_train, y_)
                logger.info('Predicting in X_train..')
                y_prediction = model.predict(X_train, y_train)
                # Based on the predictions, create PNU labels
                y_pnu = []
                for i in range(len(y_)):
                    if y_[i] == 1.0:
                        # Then this was a labeled positive
                        y_pnu.append(y_[i])
                    elif y_[i] == 0.0:
                        # This was an originally unlabeled. Check the prediction
                        if y_prediction[i] == 1.0:
                            # Then it was predicted as an unreliable negative. Mark as unlabeled (label=-1)
                            y_pnu.append(-1.0)
                        else:
                            # Then it is a reliable negative. Keep it as negative
                            y_pnu.append(y_[i])
            elif model_name == 'dif':
                model = DIF(hidden_activation='tanh', contamination=0.01)
                model.fit(X_train)
                # This decision function returns positive scores
                S1 = model.decision_function(X_train)
                max_anomaly_score = max(S1)
                min_anomaly_score = min(S1)
                mean_anomaly_score = np.mean(S1)
                logger.info(f'Max anomaly score: {max_anomaly_score}')
                logger.info(f'Min anomaly score: {min_anomaly_score}')
                logger.info(f'Mean anomaly score: {mean_anomaly_score}')
                std_score = np.std(S1)
                # Based on the predictions, create PNU labels

                # ALTERNATIVE: Keep the top outliers based on class prior
                anomaly_percentage = loss_prior
                num_outliers_to_keep = int(len(S1) * anomaly_percentage)
                # Sort anomaly scores
                anomaly_scores_idx = sorted(range(len(S1)), key=lambda k: S1[k])
                anomaly_scores_idx.reverse()
                # Keep only the top num_outliers_to_keep
                SS1 = [0] * len(S1)
                for idx in range(num_outliers_to_keep):
                    SS1[anomaly_scores_idx[idx]] = 1

                y_pnu = []
                for i in range(len(y_)):
                    if y_[i] == 1.0:
                        # Then this was a labeled positive
                        y_pnu.append(y_[i])
                    elif y_[i] == 0.0:
                        # This was an originally unlabeled. Check the prediction
                        if SS1[i] == 1:
                            #if S1[i] > mean_anomaly_score:
                            # Then it was predicted as an unreliable negative. Mark as unlabeled (label=-1)
                            y_pnu.append(-1.0)
                        else:
                            # Then it is a reliable negative. Keep it as negative
                            y_pnu.append(y_[i])
            elif model_name == 'lof':
                model = LOF(contamination=pu_ratio, n_jobs=-1)
                model.fit(X_train)
                S1 = model.decision_function(X_train)
                max_anomaly_score = max(S1)
                min_anomaly_score = min(S1)
                mean_anomaly_score = np.mean(S1)
                logger.info(f'Max anomaly score: {max_anomaly_score}')
                logger.info(f'Min anomaly score: {min_anomaly_score}')
                logger.info(f'Mean anomaly score: {mean_anomaly_score}')
                std_score = np.std(S1)
                # Based on the predictions, create PNU labels
                y_pnu = []
                for i in range(len(y_)):
                    if y_[i] == 1.0:
                        # Then this was a labeled positive
                        y_pnu.append(y_[i])
                    elif y_[i] == 0.0:
                        # This was an originally unlabeled. Check the prediction
                        if S1[i] > mean_anomaly_score:
                            # Then it was predicted as an unreliable negative. Mark as unlabeled (label=-1)
                            y_pnu.append(-1.0)
                        else:
                            # Then it is a reliable negative. Keep it as negative
                            y_pnu.append(y_[i])
            elif model_name == 'optiforest':
                model = OptIForest('L2OPT', 100, 403, 20, 1)
                model.fit(X_train)
                S1 = model.decision_function(X_train)
                # For some reason the scores are negative. Have to make them positive
                S1 = -1.0 * S1
                max_anomaly_score = max(S1)
                min_anomaly_score = min(S1)
                mean_anomaly_score = np.mean(S1)
                logger.info(f'Max anomaly score: {max_anomaly_score}')
                logger.info(f'Min anomaly score: {min_anomaly_score}')
                logger.info(f'Mean anomaly score: {mean_anomaly_score}')
                std_score = np.std(S1)
                # Based on the predictions, create PNU labels

                # ALTERNATIVE: Keep the top outliers based on class prior
                anomaly_percentage = loss_prior
                num_outliers_to_keep = int(len(S1) * anomaly_percentage)
                # Sort anomaly scores
                anomaly_scores_idx = sorted(range(len(S1)), key=lambda k: S1[k])
                anomaly_scores_idx.reverse()
                # Keep only the top num_outliers_to_keep
                SS1 = [0] * len(S1)
                for idx in range(num_outliers_to_keep):
                    SS1[anomaly_scores_idx[idx]] = 1

                y_pnu = []
                for i in range(len(y_)):
                    if y_[i] == 1.0:
                        # Then this was a labeled positive
                        y_pnu.append(y_[i])
                    elif y_[i] == 0.0:
                        # This was an originally unlabeled. Check the prediction
                        if SS1[i] == 1:
                            #if S1[i] > mean_anomaly_score + (3 * std_score):
                            # Then it was predicted as an unreliable negative. Mark as unlabeled (label=-1)
                            y_pnu.append(-1.0)
                        else:
                            # Then it is a reliable negative. Keep it as negative
                            y_pnu.append(y_[i])
            elif model_name == 'iforest':
                model = IForest()
                model.fit(X_train)
                S1 = model.decision_function(X_train)
                max_anomaly_score = max(S1)
                min_anomaly_score = min(S1)
                mean_anomaly_score = np.mean(S1)
                logger.info(f'Max anomaly score: {max_anomaly_score}')
                logger.info(f'Min anomaly score: {min_anomaly_score}')
                logger.info(f'Mean anomaly score: {mean_anomaly_score}')
                std_score = np.std(S1)
                # Based on the predictions, create PNU labels
                y_pnu = []
                for i in range(len(y_)):
                    if y_[i] == 1.0:
                        # Then this was a labeled positive
                        y_pnu.append(y_[i])
                    elif y_[i] == 0.0:
                        # This was an originally unlabeled. Check the prediction
                        if S1[i] > mean_anomaly_score + 3 * std_score:
                            # Then it was predicted as an unreliable negative. Mark as unlabeled (label=-1)
                            y_pnu.append(-1.0)
                        else:
                            # Then it is a reliable negative. Keep it as negative
                            y_pnu.append(y_[i])
            elif model_name == 'eif':
                X_train = X_train.astype('double')
                F1 = iso.iForest(X_train, ntrees=100, sample_size=256, ExtensionLevel=1)
                S1 = F1.compute_paths(X_in=X_train)
                max_anomaly_score = max(S1)
                min_anomaly_score = min(S1)
                mean_anomaly_score = np.mean(S1)
                std_score = np.std(S1)
                logger.info(f'Max anomaly score: {max_anomaly_score}')
                logger.info(f'Min anomaly score: {min_anomaly_score}')
                logger.info(f'Mean anomaly score: {mean_anomaly_score}')
                logger.info(f'STD anomaly score: {std_score}')
                # Based on the predictions, create PNU labels
                y_pnu = []

                # ALTERNATIVE: Keep the top outliers based on class prior
                anomaly_percentage = loss_prior
                num_outliers_to_keep = int(len(S1) * anomaly_percentage)
                # Reverse sort anomaly scores
                anomaly_scores_idx = sorted(range(len(S1)), key=lambda k: S1[k])
                anomaly_scores_idx.reverse()
                # Keep only the top num_outliers_to_keep
                SS1 = [0] * len(S1)
                for idx in range(num_outliers_to_keep):
                    SS1[anomaly_scores_idx[idx]] = 1

                for i in range(len(y_)):
                    if y_[i] == 1.0:
                        # Then this was a labeled positive
                        y_pnu.append(y_[i])
                    elif y_[i] == 0.0:
                        # This was an originally unlabeled. Check the prediction
                        if SS1[i] == 1:
                            #if S1[i] > mean_anomaly_score + 3*std_score:
                            # Then it was predicted as an unreliable negative. Mark as unlabeled (label=-1)
                            y_pnu.append(-1.0)
                        else:
                            # Then it is a reliable negative. Keep it as negative
                            y_pnu.append(y_[i])
            else:
                raise NotImplementedError('Model not implemented.')
            logger.info(f'[1st step] Original y_train len: {len(y_train)}')
            logger.info(f'[1st step] y_pnu len: {len(y_pnu)}')
            y_pnu = np.array(y_pnu)
            pnu_statistics(y_train, y_, y_pnu)
            perc_labeled, precision_labeled, recall_labeled = pnu_data_metrics(y_train, y_pnu)
            pnu_percent_labeled_all.append(perc_labeled)
            pnu_precision_labeled_all.append(precision_labeled)
            pnu_recall_labeled_all.append(recall_labeled)
            logger.info('------- PNU DATA METRICS ------')
            logger.info(f'[1st step] Percent labeled: {perc_labeled}')
            logger.info(f'[1st step] Precision labeled: {precision_labeled}')
            logger.info(f'[1st step] Recall labeled: {recall_labeled}')
            logger.info('-------------------------------')
            dataset = ArrayDataset(X_train, y_pnu, X_test, y_test)
            train_set, test_set = dataset.get_splits()
            train_dl = DataLoader(train_set, batch_size=32, shuffle=False)
            test_dl = DataLoader(test_set, batch_size=1024, shuffle=False)
            logger.info(
                f'Training set length: {len(train_dl.dataset)} | Test set length: {len(test_dl.dataset)}')
            input_dim = train_dl.dataset.dataset.X.shape[1]
            # Define the classifier
            if classifier_name == 'mlp':
                clf = MLP(input_dim)
            elif classifier_name == 'xgb':
                loss_alpha = -1
                loss_gamma = 3
                clf = PytorchBasedGenericGradientBoost(100, 4, criterion_name, 0.1, 0.005, 100, loss_prior,
                                                       loss_alpha, loss_gamma, loss_beta, loss_eta,
                                                       reduce="mean", pnu=True)
            elif classifier_name == 'xgb_objective' or classifier_name == 'lgb_objective':
                y_train_for_xgb = np.where(y_train == -1, 0, y_pnu)
                dtrain = xgb.DMatrix(X_train, label=y_train_for_xgb)
                d_train_lgb = lgb.Dataset(X_train, label=y_train_for_xgb)
            else:
                raise NotImplementedError('Classification model not implemented')
            logger.info('Starting training..')
            if criterion_name == 'sigmoid_npu':
                criterion = SigmoidNPU(prior=loss_prior, gamma=1, beta=loss_beta, eta=loss_eta)
            elif criterion_name == 'focal_loss_npu':
                criterion = FocalLossNPU(prior=loss_prior, alpha=loss_alpha, gamma=1, beta=loss_beta,
                                         eta=loss_eta)
            else:
                criterion = None
            # Train
            if classifier_name == 'xgb':
                clf.fit(X_train, y_pnu)
            elif classifier_name == 'xgb_objective':
                custom_loss = pnu_focal_loss(y_true=y_pnu, gamma=3.0, eta=loss_eta, pi_p=loss_prior)
                params = {
                    'objective': 'binary:logitraw',
                    'verbosity': 0,
                    'n_estimators': 1000,
                    'max_depth': 0,
                    'tree_method': 'approx',
                    'learning_rate': 1
                }
                clf = xgb.train(
                    params=params,
                    dtrain=dtrain,
                    obj=custom_loss,
                    num_boost_round=10
                )
            elif classifier_name == 'lgb_objective':
                custom_loss = pnu_focal_loss(y_true=y_pnu, gamma=3.0, eta=loss_eta, pi_p=loss_prior, lgb=True)
                params = {
                    'objective': custom_loss,
                    'verbosity': 0,
                    'n_estimators': 100,
                    'max_depth': -1,
                    'learning_rate': 0.1
                }
                clf = lgb.train(
                    params=params,
                    train_set=d_train_lgb,
                    num_boost_round=100
                )
            else:
                train(train_dl, clf, criterion, max_epochs, patience, log_interval, save_model,
                      output_log_name)
            logger.info('Evaluating in test set..')
            if classifier_name == 'xgb_objective':
                dtest = xgb.DMatrix(X_test)
                acc, auc, rprecision, precision, recall, pr_area, best_f1, f1_pos, f1_neg, f1_macro, scores, plot = \
                    validate_xgb(dtest, y_test, clf)
                if prior == 'p':
                    plot.savefig(
                        f'{output}/{experiment_name}_{labeling_mechanism}_{pu_ratio}_{model_name}_iFPNU_xgb/pr_{r}.png',
                        bbox_inches='tight')
                else:
                    plot.savefig(
                        f'{output}/{experiment_name}_{labeling_mechanism}_{pu_ratio}_{model_name}_iFPNU_xgb_{prior}/pr_{r}.png',
                        bbox_inches='tight')
                # Close the PR figure, otherwise figures pile up when running all the datasets
                plot.close()
            elif classifier_name == 'lgb_objective':
                dtest = xgb.DMatrix(X_test)
                acc, auc, rprecision, precision, recall, pr_area, best_f1, f1_pos, f1_neg, f1_macro, scores, plot = \
                    validate_xgb(X_test, y_test, clf)
                if prior == 'p':
                    plot.savefig(
                        f'{output}/{experiment_name}_{labeling_mechanism}_{pu_ratio}_{model_name}_iFPNU_lgb/pr_{r}.png',
                        bbox_inches='tight')
                else:
                    plot.savefig(
                        f'{output}/{experiment_name}_{labeling_mechanism}_{pu_ratio}_{model_name}_iFPNU_lgb_{prior}/pr_{r}.png',
                        bbox_inches='tight')
                # Close the PR figure, otherwise figures pile up when running all the datasets
                plot.close()

            else:
                acc, auc, rprecision, precision, recall, pr_area = validate(test_dl, clf)
            logger.info(f'ACCURACY: {acc}')
            logger.info(f'AUC: {auc}')
            logger.info(f'R-precision: {rprecision}')
            logger.info(f'PRECISION: {precision}')
            logger.info(f'RECALL: {recall}')
            logger.info(f'Area under PR: {pr_area}')
            logger.info(f'Best F1: {best_f1}')
            logger.info(f'F1@0.5 (pos): {f1_pos}')
            logger.info(f'F1@0.5 (neg): {f1_neg}')
            logger.info(f'F1@0.5 (macro): {f1_macro}')
            # F1 when the top prior fraction of the test scores is predicted positive
            f1_pos_prior, f1_neg_prior, f1_macro_prior, threshold = f1_at_prior_threshold(y_test, scores, loss_prior)
            logger.info(f'Prior threshold: {threshold}')
            logger.info(f'F1@prior (pos): {f1_pos_prior}')
            logger.info(f'F1@prior (neg): {f1_neg_prior}')
            logger.info(f'F1@prior (macro): {f1_macro_prior}')
            # Save the test scores, so other thresholds or metrics can be computed without re-running
            scores_dir = f'{output}/{experiment_name}_{labeling_mechanism}_{pu_ratio}_{model_name}_iFPNU_lgb'
            scores_dir += '' if prior == 'p' else f'_{prior}'
            np.savez(f'{scores_dir}/scores_{r}.npz', y_test=np.ravel(y_test), scores=scores, threshold=threshold,
                     prior=loss_prior)
            # Store individual results
            acc_all.append(acc)
            auc_all.append(auc)
            rprecision_all.append(rprecision)
            precision_all.append(precision)
            recall_all.append(recall)
            pr_area_all.append(pr_area)
            f1_all.append(best_f1)
            f1_pos_all.append(f1_pos)
            f1_neg_all.append(f1_neg)
            f1_macro_all.append(f1_macro)
            f1_pos_prior_all.append(f1_pos_prior)
            f1_neg_prior_all.append(f1_neg_prior)
            f1_macro_prior_all.append(f1_macro_prior)
        # Average and summarize results
        results = average_results(acc_all, auc_all, rprecision_all, precision_all, recall_all,
                                  pr_area_all, f1_all, f1_pos_all, f1_neg_all, f1_macro_all,
                                  f1_pos_prior_all, f1_neg_prior_all, f1_macro_prior_all)
        logger.info(f'===========================================================')
        logger.info('')
        logger.info(f'Overall results')
        logger.info(f'All Accuracy: {acc_all}')
        logger.info(f'All AUC: {auc_all}')
        logger.info(f'All R-precision: {rprecision_all}')
        logger.info(f'All Precision: {precision_all}')
        logger.info(f'All Recall: {recall_all}')
        logger.info(f'All PR_area: {pr_area_all}')
        logger.info(f'All F1: {f1_all}')
        logger.info(f'All F1@0.5 (pos): {f1_pos_all}')
        logger.info(f'All F1@0.5 (neg): {f1_neg_all}')
        logger.info(f'All F1@0.5 (macro): {f1_macro_all}')
        logger.info(f'All F1@prior (pos): {f1_pos_prior_all}')
        logger.info(f'All F1@prior (neg): {f1_neg_prior_all}')
        logger.info(f'All F1@prior (macro): {f1_macro_prior_all}')
        logger.info(f'--------------------------')
        logger.info(f'Average results')
        logger.info(
            f'ACC: {results["mean_acc"]:.2f} | Std: {results["stdev_acc"]:.2f} | StdErr: {results["stder_acc"]:.2f}')
        logger.info(
            f'AUC: {results["mean_auc"]:.2f} | Std: {results["stdev_auc"]:.2f} | StdErr: {results["stder_auc"]:.2f}')
        logger.info(
            f'R-p: {results["mean_rprecision"]:.2f} | Std: {results["stdev_rprecision"]:.2f} | StdErr: {results["stder_rprecision"]:.2f}')
        logger.info(
            f'P: {results["mean_precision"]:.2f} | Std: {results["stdev_precision"]:.2f} | StdErr: {results["stder_precision"]:.2f}')
        logger.info(
            f'R: {results["mean_recall"]:.2f} | Std: {results["stdev_recall"]:.2f} | StdErr: {results["stder_recall"]:.2f}')
        logger.info(
            f'AUPR: {results["mean_pr_area"]:.2f} | Std: {results["stdev_pr_area"]:.2f} | StdErr: {results["stder_pr_area"]:.2f}')
        logger.info(
            f'F1: {results["mean_f1"]:.2f} | Std: {results["stdev_f1"]:.2f} | StdErr: {results["stder_f1"]:.2f}')
        logger.info(
            f'F1-pos@0.5: {results["mean_f1_pos"]:.2f} | Std: {results["stdev_f1_pos"]:.2f} | StdErr: {results["stder_f1_pos"]:.2f}')
        logger.info(
            f'F1-neg@0.5: {results["mean_f1_neg"]:.2f} | Std: {results["stdev_f1_neg"]:.2f} | StdErr: {results["stder_f1_neg"]:.2f}')
        logger.info(
            f'F1-macro@0.5: {results["mean_f1_macro"]:.2f} | Std: {results["stdev_f1_macro"]:.2f} | StdErr: {results["stder_f1_macro"]:.2f}')
        logger.info(
            f'F1-pos@prior: {results["mean_f1_pos_prior"]:.2f} | Std: {results["stdev_f1_pos_prior"]:.2f} | StdErr: {results["stder_f1_pos_prior"]:.2f}')
        logger.info(
            f'F1-neg@prior: {results["mean_f1_neg_prior"]:.2f} | Std: {results["stdev_f1_neg_prior"]:.2f} | StdErr: {results["stder_f1_neg_prior"]:.2f}')
        logger.info(
            f'F1-macro@prior: {results["mean_f1_macro_prior"]:.2f} | Std: {results["stdev_f1_macro_prior"]:.2f} | StdErr: {results["stder_f1_macro_prior"]:.2f}')

        pnu_metrics_avg_results = average_pnu_data_metrics_results(pnu_percent_labeled_all,
                                                                   pnu_precision_labeled_all,
                                                                   pnu_recall_labeled_all)
        logger.info(f'--------------------------')
        logger.info(f'Average PNU data METRICS results')
        logger.info(
            f'Percent Labeled: {pnu_metrics_avg_results["mean_percent"]:.2f} | Std: {pnu_metrics_avg_results["stdev_percent"]:.2f}')
        logger.info(
            f'Precision Labeled: {pnu_metrics_avg_results["mean_precision"]:.2f} | Std: {pnu_metrics_avg_results["stdev_precision"]:.2f}')
        logger.info(
            f'Recall Labeled: {pnu_metrics_avg_results["mean_recall"]:.2f} | Std: {pnu_metrics_avg_results["stdev_recall"]:.2f}')

        all_experimental_results[experiment_name] = {
            'ACC': results["mean_acc"],
            'AUC': results["mean_auc"],
            'R-p': results["mean_rprecision"],
            'P': results["mean_precision"],
            'R': results["mean_recall"],
            'AUPR': results["mean_pr_area"],
            'F1': results["mean_f1"],
        }
        all_experimental_PNU_data_results[experiment_name] = {
            'PERCENT': pnu_metrics_avg_results["mean_percent"],
            'PRECISION': pnu_metrics_avg_results["mean_precision"],
            'RECALL': pnu_metrics_avg_results["mean_recall"]

        }
        res = pd.DataFrame()
        runs = [i for i in range(num_runs)]
        labeling = [labeling_mechanism] * num_runs
        dataset = [experiment_name] * num_runs
        pu_rate = [pu_ratio] * num_runs
        res['run'] = runs
        res['dataset'] = dataset
        res['pu_ratio'] = pu_rate
        res['labeling_mechanism'] = labeling
        res['roc_auc'] = auc_all
        res['ap'] = pr_area_all
        res['f1'] = f1_all
        res['f1_pos_05'] = f1_pos_all
        res['f1_neg_05'] = f1_neg_all
        res['f1_macro_05'] = f1_macro_all
        res['f1_pos_prior'] = f1_pos_prior_all
        res['f1_neg_prior'] = f1_neg_prior_all
        res['f1_macro_prior'] = f1_macro_prior_all
        if prior == 'p':
            res.to_csv(
                f'{output}/{experiment_name}_{labeling_mechanism}_{pu_ratio}_{model_name}_iFPNU_lgb/results.csv',
                index=False)
        else:
            res.to_csv(
                f'{output}/{experiment_name}_{labeling_mechanism}_{pu_ratio}_{model_name}_iFPNU_lgb_{prior}/results.csv',
                index=False)

        logger.handlers.clear()
    print('\n==============')
    print('All experimental results:')
    print(f'Experiment\t\t\tAUC\t\tAUPR')
    for experiment in all_experimental_results:
        auc = all_experimental_results[experiment]['AUC']
        aupr = all_experimental_results[experiment]['AUPR']
        print(f'{experiment}\t{auc:.2f}\t{aupr:.2f}')
    print('\n==============')
    print('All experimental PNU data metrics results:')
    print(f'Experiment\t\tPERCENT\t\tPRECISION\t\tRECALL_RELIABLE')
    for experiment in all_experimental_PNU_data_results:
        percent = all_experimental_PNU_data_results[experiment]['PERCENT']
        precision = all_experimental_PNU_data_results[experiment]['PRECISION']
        recall = all_experimental_PNU_data_results[experiment]['RECALL']
        print(f'{experiment}\t{percent:.2f}\t{precision:.2f}\t{recall:.2f}')
