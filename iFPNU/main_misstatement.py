#
# main_misstatement.py
#

import sys
import logging
import configparser
import torch
import torch.nn as nn
import torch.optim as optim
from data_loaders.json_loader import JSONDataset, get_unique_categorical_counts, get_categ_cont_target_values
from models.gated_tab_transformer import GatedTabTransformer
from pu.losses import SampleDependentFocalLossNNPU, SigmoidNNPU
from train_and_validation import train_gtt, validate_gtt


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


def get_data(path, labelsdir):
    """
    Get the data
    :param: path the directory to the dataset
    :return: train and test data
    """
    dataset = JSONDataset(path, labelsdir)
    # Get train and test data
    train_df, test_df = dataset.get_dataframes()
    return train_df, test_df


if __name__ == "__main__":
    # Read configuration file
    config = configparser.ConfigParser()
    config.read('./config.ini')
    # Running configuration
    experiment_name = config['RUN_MISSTATEMENT']['EXPERIMENT']
    criterion_name = config['RUN_MISSTATEMENT']['CRITERION']
    model_name = config['RUN_MISSTATEMENT']['MODEL']
    dirpath = config['MISSTATEMENT'][experiment_name]
    labelsdir = config['MISSTATEMENT']['LABELSDIR']
    # Loss parameters configuration
    loss_prior = float(config['RUN_MISSTATEMENT']['POS_PRIOR'])
    loss_alpha = float(config['LOSS_PARAMS']['ALPHA'])
    loss_gamma = float(config['LOSS_PARAMS']['GAMMA'])
    loss_beta = float(config['LOSS_PARAMS']['BETA'])
    device_type = config['RUN_MISSTATEMENT']['DEVICE']
    device = torch.device(device_type)
    output_log_name = f'{experiment_name}_{criterion_name}_{model_name}'

    loss_prior = 1 - loss_prior
    # Set up logging
    logFormatter = logging.Formatter(
        "%(asctime)s |%(name)s| [%(levelname)-5.5s]  %(message)s")
    logger = logging.getLogger('Main')
    fileHandler = logging.FileHandler(
        "{0}/{1}.log".format('output', output_log_name))
    fileHandler.setFormatter(logFormatter)
    logger.addHandler(fileHandler)
    consoleHandler = logging.StreamHandler(stream=sys.stdout)
    consoleHandler.setFormatter(logFormatter)
    logger.addHandler(ColorHandler())
    logger.setLevel(logging.INFO)

    logger.info(f'EXPERIMENT: {experiment_name}')
    logger.info(f'Model: {model_name} | {criterion_name}')
    logger.info(f'===========================================================')

    train_dataframe, test_dataframe = get_data(dirpath, labelsdir)
    n_categories = get_unique_categorical_counts(train_dataframe, 42)
    train_cont, train_categ, train_target = get_categ_cont_target_values(train_dataframe, 1, 42)
    test_cont, test_categ, test_target = get_categ_cont_target_values(test_dataframe, 1, 42)
    num_continuous = train_cont.shape[1]
    # Define criterion
    if criterion_name == 'sample_dependent_focal_loss_nnpu':
        criterion = SampleDependentFocalLossNNPU(prior=loss_prior, alpha=loss_alpha, gamma=loss_gamma,
                                                 beta=loss_beta, device=device)
    elif criterion_name == 'sigmoid_nnpu':
        criterion = SigmoidNNPU(prior=loss_prior, gamma=1, beta=loss_beta)
    else:
        criterion = None
    # define model
    model = GatedTabTransformer(
        categories=n_categories,
        num_continuous=num_continuous,
        transformer_dim=32,
        dim_out=1,
        transformer_depth=6,
        transformer_heads=8,
        attn_dropout=0.0,
        ff_dropout=0.0,
        mlp_act=nn.LeakyReLU(0.05),
        mlp_depth=2,
        mlp_dimension=8,
        gmlp_enabled=True,
    )
    model = model.train().to(device=device)
    # Define optimizer
    optimizer = optim.AdamW(params=model.parameters(), lr=1e-3, amsgrad=True)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    train_gtt(model,
              criterion,
              optimizer,
              scheduler,
              train_cont,
              train_categ,
              train_target,
              test_cont,
              test_categ,
              test_target,
              device=device,
              batch_size=256,
              max_epochs=100,
              patience=5,
              save_best_model_dict=True,
              log_interval=10,
              exp=experiment_name
              )
    logger.info('Evaluating in test set..')
    acc, auc, rprecision, precision, recall, pr_area = validate_gtt(model, test_cont, test_categ, test_target, device=device)
    logger.info(f'ACCURACY: {acc}')
    logger.info(f'AUC: {auc}')
    logger.info(f'R-precision: {rprecision}')
    logger.info(f'PRECISION: {precision}')
    logger.info(f'RECALL: {recall}')
    logger.info(f'Area under PR: {pr_area}')