# main_DIF_ablation
# Just run the first step alone, i.e., the anomaly detection module as an ablation example

from data_loaders.csv_loader import CSVDataset
from torch.utils.data import DataLoader
from data_loaders.pu_data import create_pu_data
import configparser
import sys
from pyod.models.dif import DIF
import pandas as pd
import torch
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score, average_precision_score
from pu.evaluation import r_precision_at_n
from sklearn.metrics import precision_score, precision_recall_curve
from sklearn.metrics import recall_score
import logging
import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')  # Suppress all warnings


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
    train_dl = DataLoader(train, batch_size=32, shuffle=False, pin_memory=False)
    test_dl = DataLoader(test, batch_size=1024, shuffle=False, pin_memory=False)
    return train_dl, test_dl


def prepare_pu_data(path, pu_ratio, labeling_mechanism, random_state):
    """
    Prepare data: load the dataset, create splits, create PU data in training set
    and return the corresponding data loaders
    :param path: path to dataset
    :param pu_ratio: the ratio of the positive labeled examples
    :param labeling_mechanism: the labeling mechanism assumption
    :param random_state: random state for splitting
    :return: train and test data
    """
    # Load the dataset
    dataset = CSVDataset(path)
    # Get split
    train, test = dataset.get_splits(random_state=random_state)
    # Create PU data
    x_train = train.dataset.X[train.indices]
    y_train = train.dataset.y[train.indices]
    loss_prior = np.mean(y_train)
    y_ = create_pu_data(x_train, y_train, pu_ratio, labeling_mechanism, None)
    # Set y_ to dataset
    train.dataset.update_y(y_)
    # Prepare data loaders
    train_dl = DataLoader(train, batch_size=32, shuffle=False, pin_memory=False)
    test_dl = DataLoader(test, batch_size=1024, shuffle=False, pin_memory=False)
    return train_dl, test_dl, loss_prior


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
    # Load the dataset
    dataset = CSVDataset(path)
    # Get split
    train, test = dataset.get_splits(random_state=random_state)
    # Create PU data
    x_train = train.dataset.X[train.indices]
    y_train = train.dataset.y[train.indices]
    loss_prior = np.mean(y_train)
    x_test = test.dataset.X[test.indices]
    y_test = test.dataset.y[test.indices]
    y_ = create_pu_data(x_train, y_train, pu_ratio, labeling_mechanism, None)
    return x_train, y_, x_test, y_test, loss_prior


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


if __name__ == "__main__":
    # Read configuration file
    config = configparser.ConfigParser()
    config.read('./config.ini')
    experiment_name = config['RUN']['EXPERIMENT']
    output = config['RUN']['OUTPUT']
    model_name = 'DIF'
    criterion_name = 'default'
    max_epochs = int(config['RUN']['MAX_EPOCHS'])
    patience = int(config['RUN']['PATIENCE'])
    log_interval = int(config['RUN']['LOG_INTERVAL'])
    save_model = config['RUN'].getboolean('SAVE_BEST_MODEL')
    device_type = config['RUN']['DEVICE']
    dataset_path = config['DATASETS'][experiment_name]
    pu_ratio = float(config['RUN']['PU_RATIO'])
    labeling_mechanism = config['RUN']['LABELING_MECHANISM']
    num_runs = int(config['RUN']['NUM_RUNS'])
    log_wandb = config['RUN'].getboolean('LOG_WANDB')
    # Loss parameters configuration
    loss_gamma = float(config['LOSS_PARAMS']['GAMMA'])
    loss_beta = float(config['LOSS_PARAMS']['BETA'])
    prior = config['RUN']['PRIOR']
    device = torch.device(device_type)
    if prior == 'p':
        output_log_name = f'{experiment_name}_{pu_ratio}_{labeling_mechanism}_{num_runs}_' \
                          f'{criterion_name}_{model_name}'
    else:
        output_log_name = f'{experiment_name}_{pu_ratio}_{labeling_mechanism}_{num_runs}_' \
                          f'{criterion_name}_{model_name}_{prior}'
    # Set up logging
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
        logger.info(f'>>>>>>>>>> Run {i + 1} <<<<<<<<<<')
        train_dl, test_dl, loss_prior = prepare_pu_data(dataset_path, pu_ratio, labeling_mechanism,
                                                        random_state=i)
        # The following line is needed in the case of Gradient Boosting that takes as input the np matrices
        X_train, y_train, X_test, y_test, loss_prior = prepare_pu_data_plain(dataset_path, pu_ratio,
                                                                             labeling_mechanism,
                                                                             random_state=i)
        y_train = y_train.flatten()
        y_test = y_test.astype(int)
        logger.info(
            f'Training set length: {len(train_dl.dataset)} | Test set length: {len(test_dl.dataset)}')
        input_dim = train_dl.dataset.dataset.X.shape[1]

        # Define model
        model = DIF(hidden_activation='tanh', contamination=0.01)
        # Train
        logger.info('Starting training..')
        model.fit(X_train)
        # Test
        logger.info('Evaluating in test set..')
        y_test_pred = model.predict(X_test)  # outlier labels (0 or 1)
        y_test_scores = model.decision_function(X_test)  # outlier scores

        # Calculate scores
        acc = accuracy_score(y_test, y_test_pred)
        auc = roc_auc_score(y_test, y_test_scores)
        rprecision = r_precision_at_n(y_test_scores, y_test)
        precision = precision_score(y_test, y_test_pred, average='binary')
        recall = recall_score(y_test, y_test_pred, average='binary')
        pr_area = average_precision_score(y_test, y_test_scores)
        # Plot PR curves
        p, r, thresholds = precision_recall_curve(y_test, y_test_scores)
        # Find the threshold that gives the best F1-score
        numerator = 2 * r * p
        denom = r + p
        f1_scores = np.divide(numerator, denom, out=np.zeros_like(denom), where=(denom != 0))
        best_f1 = np.max(f1_scores)
        best_threshold_index = np.argmax(f1_scores)
        best_threshold = thresholds[best_threshold_index]
        print(f"Best Threshold: {best_threshold:.2f}")
        print(f"Best F1-score: {best_f1:.2f}")
        # Plot the precision-recall curve
        plt.figure(figsize=(8, 6))
        plt.plot(r, p, label='Precision-Recall curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc='best')
        # Plot the threshold points
        plt.scatter(r[best_threshold_index], p[best_threshold_index], color='red',
                    label=f'Best Threshold (F1 = {best_f1:.2f}, AP = {pr_area:.2f})')
        plt.legend(loc='best')

        if prior == 'p':
            plt.savefig(f'{output}/{experiment_name}_{labeling_mechanism}_{pu_ratio}_default_DIF/pr_{i}.png',
                        bbox_inches='tight')
        else:
            plt.savefig(
                f'{output}/{experiment_name}_{labeling_mechanism}_{pu_ratio}_default_DIF_{prior}/pr_{i}.png',
                bbox_inches='tight')
        logger.info(f'ACCURACY: {acc}')
        logger.info(f'AUC: {auc}')
        logger.info(f'R-precision: {rprecision}')
        logger.info(f'PRECISION: {precision}')
        logger.info(f'RECALL: {recall}')
        logger.info(f'Area under PR: {pr_area}')
        logger.info(f'Best F1: {best_f1}')
        # Store individual results
        acc_all.append(acc)
        auc_all.append(auc)
        rprecision_all.append(rprecision)
        precision_all.append(precision)
        recall_all.append(recall)
        pr_area_all.append(pr_area)
        f1_all.append(best_f1)

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
    if prior == 'p':
        res.to_csv(f'{output}/{experiment_name}_{labeling_mechanism}_{pu_ratio}_default_DIF/results.csv',
                   index=False)
    else:
        res.to_csv(
            f'{output}/{experiment_name}_{labeling_mechanism}_{pu_ratio}_default_DIF_{prior}/results.csv',
            index=False)
