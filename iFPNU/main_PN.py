#
# main_PN.py
# Rub XGBoost assuming PN learning
#

import configparser
from data_loaders.csv_loader import CSVDataset
import numpy as np
import logging
import wandb
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
from pu.evaluation import r_precision_at_n
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import sys
from pu.fpu_xgb import FLLoss
import lightgbm as lgb


def prepare_data(path, random_state):
    """
    Prepare data: load the dataset, create splits (assume PN classification)
    :param path: path to dataset
    :return: train and test data
    """
    # Load the dataset
    dataset = CSVDataset(path)
    # Get split
    train, test = dataset.get_splits(random_state=random_state)
    # Prepare data loaders
    x_train = train.dataset.X[train.indices]
    y_train = train.dataset.y[train.indices]
    x_test = test.dataset.X[test.indices]
    y_test = test.dataset.y[test.indices]
    return x_train, y_train, x_test, y_test


def average_results(accuracies, aucs, rprecisions, precisions, recalls, pr_areas, f1):
    """
    Average individual results
    :param accuracies: list of accuracy results
    :param aucs: list of auc results
    :param rprecisions: list of rprecision results
    :param precisions: list of precision results
    :param recalls: list of recall results
    :param pr_areas: list of pr_area results
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
        'stder_f1': np.std(f1) / np.sqrt(np.size(f1))
    }
    return avg_results


def validate_xgb(x_test, y_test, model):
    """
    Validate the xgb model
    """
    pred_proba = model.predict_proba(x_test)[:, 1]
    predictions = []
    for y_hat in pred_proba:
        yhat_round = y_hat.round()
        predictions.append(yhat_round)
    # Calculate accuracy
    acc = accuracy_score(y_test, predictions)
    # Calculate AUC
    area = roc_auc_score(y_test, pred_proba)
    # Calculate R-precision
    rprecision = r_precision_at_n(pred_proba, y_test)
    # Calculate precision and recall
    precision = precision_score(y_test, predictions, average='binary')
    recall = recall_score(y_test, predictions, average='binary')
    # Calculate area under PR
    pr_area = average_precision_score(y_test, pred_proba)
    # Plot PR curves
    p, r, thresholds = precision_recall_curve(y_test, pred_proba)
    # Find the threshold that gives the best F1-score
    numerator = 2 * r * p
    denom = r + p
    f1_scores = np.divide(numerator, denom, out=np.zeros_like(denom), where=(denom != 0))
    best_f1 = np.max(f1_scores)
    best_threshold_index = np.argmax(f1_scores)
    best_threshold = thresholds[best_threshold_index]
    return acc, area, rprecision, precision, recall, pr_area, best_f1


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


if __name__ == '__main__':
    # Read configuration file
    config = configparser.ConfigParser()
    config.read('./config.ini')
    # Running configuration
    experiment_name = config['RUN']['EXPERIMENT']
    model_name = config['RUN']['MODEL']
    criterion_name = config['RUN']['CRITERION']
    max_epochs = int(config['RUN']['MAX_EPOCHS'])
    patience = int(config['RUN']['PATIENCE'])
    log_interval = int(config['RUN']['LOG_INTERVAL'])
    save_model = config['RUN'].getboolean('SAVE_BEST_MODEL')
    device_type = config['RUN']['DEVICE']
    dataset_path = config['DATASETS'][experiment_name]
    pu_ratio = float(config['RUN']['PU_RATIO'])
    labeling_mechanism = config['RUN']['LABELING_MECHANISM']
    num_runs = int(config['RUN']['NUM_RUNS'])
    output_log_name = f'{experiment_name}_{pu_ratio}_{labeling_mechanism}_{num_runs}_' \
                      f'{criterion_name}_{model_name}'
    log_wandb = config['RUN'].getboolean('LOG_WANDB')
    # Loss parameters configuration
    loss_alpha = float(config['LOSS_PARAMS']['ALPHA'])
    # loss_alpha = 2.0*loss_alpha
    loss_gamma = float(config['LOSS_PARAMS']['GAMMA'])
    loss_beta = float(config['LOSS_PARAMS']['BETA'])
    #loss_prior = float(config['DATASET_POS_PRIORS'][experiment_name])
    # Set up logging
    logFormatter = logging.Formatter(
        "%(asctime)s |%(name)s| [%(levelname)-5.5s]  %(message)s")
    logger = logging.getLogger('Main')
    fileHandler = logging.FileHandler(
        "{0}/{1}.log".format('output_PN', output_log_name))
    fileHandler.setFormatter(logFormatter)
    logger.addHandler(fileHandler)
    consoleHandler = logging.StreamHandler(stream=sys.stdout)
    consoleHandler.setFormatter(logFormatter)
    logger.addHandler(ColorHandler())
    logger.setLevel(logging.INFO)

    # Setup variables for holding results
    acc_all = []
    auc_all = []
    rprecision_all = []
    precision_all = []
    recall_all = []
    pr_area_all = []
    f1_all = []

    logger.info(f'EXPERIMENT: {experiment_name}')
    logger.info(f'PU_RATIO: {pu_ratio} {labeling_mechanism} | Runs: {num_runs} | '
                f'Model: {model_name} | {criterion_name}')
    logger.info(f'===========================================================')
    for i in range(num_runs):
        run = None
        if log_wandb:
            # Setup wandb
            run = wandb.init(
                # set the wandb project where this run will be logged
                project=f'{experiment_name} | {pu_ratio} {labeling_mechanism} | {model_name} | {criterion_name}',
                name=f"Run_{i + 1}",
                # track hyperparameters and run metadata
                config={
                    "criterion": criterion_name,
                    "architecture": model_name,
                    "dataset": experiment_name,
                    "epochs": max_epochs,
                    "alpha": loss_alpha,
                    "gamma": loss_gamma
                }
            )

        logger.info(f'>>>>>>>>>> Run {i + 1} <<<<<<<<<<')

        X_train, y_train, X_test, y_test = prepare_data(dataset_path, random_state=i)
        logger.info(
            f'Training set length: {len(X_train)} | Test set length: {len(X_test)}')
        input_dim = X_train.shape[1]
        # Define the network
        #model = xgb.XGBClassifier(n_estimators=100, max_depth=5, learning_rate=1, objective='binary:logistic')
        model = xgb.XGBClassifier(n_estimators=100, max_depth=5, tree_method="auto",
                                  objective=FLLoss(r1=2))
        #model = lgb.LGBMClassifier(learning_rate=0.1, num_boost_round=10, n_estimators=100, max_depth=5)
        model.fit(X_train, y_train)
        logger.info('Evaluating in test set..')
        acc, auc, rprecision, precision, recall, pr_area, f1 = validate_xgb(X_test, y_test, model)

        logger.info(f'ACCURACY: {acc}')
        logger.info(f'AUC: {auc}')
        logger.info(f'R-precision: {rprecision}')
        logger.info(f'PRECISION: {precision}')
        logger.info(f'RECALL: {recall}')
        logger.info(f'Area under PR: {pr_area}')
        logger.info(f'Best F1: {f1}')
        # Store individual results
        acc_all.append(acc)
        auc_all.append(auc)
        rprecision_all.append(rprecision)
        precision_all.append(precision)
        recall_all.append(recall)
        pr_area_all.append(pr_area)
        f1_all.append(f1)
        if log_wandb:
            # Finish run to sync with wandb
            run.finish()
    # Average and summarize results
    results = average_results(acc_all, auc_all, rprecision_all, precision_all, recall_all,
                              pr_area_all, f1_all)
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
