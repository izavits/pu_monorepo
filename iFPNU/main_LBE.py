#
# main_LBE.py
#

import configparser
import numpy as np
import pandas as pd
import torch
from data_loaders.csv_loader import CSVDataset
from data_loaders.pu_data import create_pu_data
from torch.utils.data import DataLoader
from train_and_validation import validate_sarem, f1_at_prior_threshold
import logging
import sys
import wandb
from models.LBE import LBE, LBEdeep


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
    train_dl = DataLoader(train, batch_size=32, shuffle=False)
    test_dl = DataLoader(test, batch_size=1024, shuffle=False)
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
    train_dl = DataLoader(train, batch_size=32, shuffle=False)
    test_dl = DataLoader(test, batch_size=1024, shuffle=False)
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
    # Running configuration
    experiment_name = config['RUN']['EXPERIMENT']
    output = config['RUN']['OUTPUT']
    model_name = 'LBE'
    criterion_name = 'default'
    max_epochs = int(config['RUN']['MAX_EPOCHS'])
    patience = int(config['RUN']['PATIENCE'])
    log_interval = int(config['RUN']['LOG_INTERVAL'])
    save_model = config['RUN'].getboolean('SAVE_BEST_MODEL')
    device_type = config['RUN']['DEVICE']
    pu_ratio = float(config['RUN']['PU_RATIO'])
    labeling_mechanism = config['RUN']['LABELING_MECHANISM']
    num_runs = int(config['RUN']['NUM_RUNS'])
    log_wandb = config['RUN'].getboolean('LOG_WANDB')
    run_all = config['RUN'].getboolean('RUN_ALL', fallback=False)
    # Loss parameters configuration
    loss_gamma = float(config['LOSS_PARAMS']['GAMMA'])
    loss_beta = float(config['LOSS_PARAMS']['BETA'])
    prior = config['RUN']['PRIOR']
    device = torch.device(device_type)

    # Run all the datasets of the [DATASETS] section, or only the configured experiment
    EXPERIMENTS = []
    if run_all:
        for key in config['DATASETS'].keys():
            EXPERIMENTS.append(key.upper())
    else:
        EXPERIMENTS.append(experiment_name)

    # Set up logging (console output is shared, each dataset gets its own log file)
    logFormatter = logging.Formatter(
        "%(asctime)s |%(name)s| [%(levelname)-5.5s]  %(message)s")
    logger = logging.getLogger('Main')
    logger.addHandler(ColorHandler())
    logger.setLevel(logging.INFO)

    for experiment_name in EXPERIMENTS:
        dataset_path = config['DATASETS'][experiment_name]
        if prior == 'p':
            output_log_name = f'{experiment_name}_{pu_ratio}_{labeling_mechanism}_{num_runs}_' \
                              f'{criterion_name}_{model_name}'
        else:
            output_log_name = f'{experiment_name}_{pu_ratio}_{labeling_mechanism}_{num_runs}_' \
                              f'{criterion_name}_{model_name}_{prior}'
        fileHandler = logging.FileHandler(
            "{0}/{1}.log".format(output, output_log_name))
        fileHandler.setFormatter(logFormatter)
        logger.addHandler(fileHandler)

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
                        #"alpha": loss_alpha,
                        "gamma": loss_gamma
                    }
                )

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
            # Define the network
            model = LBE()
            logger.info('Starting training..')
            model.fit(X_train, y_train)
            logger.info('Evaluating in test set..')
            acc, auc, rprecision, precision, recall, pr_area, best_f1, f1_pos, f1_neg, f1_macro, scores, plot = \
                validate_sarem(X_test, y_test, model)
            if prior == 'p':
                plot.savefig(f'{output}/{experiment_name}_{labeling_mechanism}_{pu_ratio}_default_LBE/pr_{i}.png',
                             bbox_inches='tight')
            else:
                plot.savefig(
                        f'{output}/{experiment_name}_{labeling_mechanism}_{pu_ratio}_default_LBE_{prior}/pr_{i}.png',
                        bbox_inches='tight')
            # Close the PR figure, otherwise figures pile up when running all the datasets
            plot.close()

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
            scores_dir = f'{output}/{experiment_name}_{labeling_mechanism}_{pu_ratio}_default_LBE'
            scores_dir += '' if prior == 'p' else f'_{prior}'
            np.savez(f'{scores_dir}/scores_{i}.npz', y_test=np.ravel(y_test), scores=scores, threshold=threshold,
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
            if log_wandb:
                # Finish run to sync with wandb
                run.finish()
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

        res = pd.DataFrame()
        runs = [i for i in range(num_runs)]
        labeling = [labeling_mechanism]*num_runs
        dataset = [experiment_name]*num_runs
        pu_rate = [pu_ratio]*num_runs
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
            res.to_csv(f'{output}/{experiment_name}_{labeling_mechanism}_{pu_ratio}_default_LBE/results.csv',
                       index=False)
        else:
            res.to_csv(f'{output}/{experiment_name}_{labeling_mechanism}_{pu_ratio}_default_LBE_{prior}/results.csv',
                       index=False)

        # Close this dataset's log file before moving to the next one
        logger.removeHandler(fileHandler)
        fileHandler.close()
