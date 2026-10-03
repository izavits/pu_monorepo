#
# json_loader.py
# Dataset loader for the JSON files
#

import json
from torch.utils.data import Dataset
import pandas as pd
import numpy as np


class JSONDataset(Dataset):
    def __init__(self, dirpath, labelsdir):
        """
        Load the dataset from JSON as a dataframe.
        Store the inputs and outputs and label encode outputs
        :param dirpath: the directory where the train and test ids are
        :param labelsdir: the directory where the labels are
        """
        training_positives = []
        training_negatives = []
        test_positives = []
        test_negatives = []
        feature_list = [
            'act', 'ap', 'at', 'ceq', 'che', 'cogs', 'csho', 'dlc', 'dltis', 'dltt',
            'dp', 'ib', 'invt', 'ivao', 'ivst', 'lct', 'lt', 'ni', 'ppegt', 'pstk',
            're', 'rect', 'sale', 'sstk', 'txp', 'txt', 'xint', 'prcc_f',
            'dch_wc', 'ch_rsst', 'dch_rec', 'dch_inv', 'soft_assets', 'ch_cs',
            'ch_cm', 'ch_roa', 'issue', 'bm', 'dpi', 'reoa', 'EBIT', 'ch_fcf'
        ]
        with open(dirpath + '/train_ids.json', 'r') as fp:
            train_split = json.load(fp)
        with open(dirpath + '/test_ids.json', 'r') as fp:
            test_split = json.load(fp)
        with open(labelsdir + '/labels_merged_BAO_with_reasons_Over_100words_R.json', 'r') as fp:
            labels = json.load(fp)
        for instance in labels:
            basename = instance['Basename']
            label = instance['Misstatement_Reliable']
            features = []
            industry = instance['SIC']
            state = instance['State']
            features.append(industry)
            features.append(state)
            for feat in feature_list:
                features.append(float(instance[feat]))
            if basename in train_split:
                if label == 1:
                    training_positives.append(features)
                else:
                    training_negatives.append(features)
            if basename in test_split:
                if label == 1:
                    test_positives.append(features)
                else:
                    test_negatives.append(features)
        all_training_examples = training_positives + training_negatives
        training_categories = [1] * len(training_positives)
        training_categories += [0] * len(training_negatives)
        all_test_examples = test_positives + test_negatives
        test_categories = [1] * len(test_positives)
        test_categories += [0] * len(test_negatives)
        # Add the class at the end of each example vector
        for i, example in enumerate(all_training_examples):
            example.append(training_categories[i])
        for i, example in enumerate(all_test_examples):
            example.append(test_categories[i])
        # Transform to dataframes
        self.train_df = pd.DataFrame(all_training_examples)
        self.test_df = pd.DataFrame(all_test_examples)
        self.X_train = self.train_df.values[:, :-1]
        self.y_train = self.train_df.values[:, -1]
        self.X_test = self.test_df.values[:, :-1]
        self.y_test = self.test_df.values[:, -1]


    def __len__(self):
        """
        Get the number of rows in the dataset
        :return: number of rows
        """
        return len(self.X_train) + len(self.X_test)

    def __getitem__(self, idx):
        """
        Get a row at a specific index
        :param idx: the specified index
        :return: the row
        """
        max_train_idx = len(self.X_train) - 1
        if idx < len(self.X_train):
            return [self.X_train[idx], self.y_train[idx]]
        else:
            return [self.X_test[idx - max_train_idx], self.y_test[idx - max_train_idx]]

    def get_splits(self):
        """
        Get the train test splits directly
        :return: the X_train, y_train, X_test, y_test
        """
        return self.X_train, self.y_train, self.X_test, self.y_test

    def get_dataframes(self):
        """
        Get the train test dataframes
        :return: the train_df and test_df
        """
        return self.train_df, self.test_df


def get_unique_categorical_counts(dataset, cont_count=10):
    return dataset.iloc[:, 0:-(cont_count + 1)].nunique().to_list()


def get_target_values(dataframe, positive_class=1):
    targets_source = dataframe.iloc[:, -1:]
    target = np.zeros(targets_source.shape)
    for i in range(targets_source.shape[0]):
        target[i] = 1 if targets_source.iloc[i, 0] == positive_class else 0
    return target


def get_cont_values(dataframe, cont_count=10):
    cont = dataframe.iloc[:, -(cont_count + 1):-1]
    return cont.to_numpy()


def get_categ_values(dataframe, cont_count=10):
    categ = dataframe.iloc[:, 0:-(cont_count + 1)]
    for i in range(categ.shape[1]):
        categ[categ.columns[i]] = categ[categ.columns[i]].astype("category").cat.codes
    return categ.to_numpy()


def get_categ_cont_target_values(dataframe, positive_class=1, cont_count=10):
    target = get_target_values(dataframe, positive_class)
    cont = get_cont_values(dataframe, cont_count)
    categ = get_categ_values(dataframe, cont_count)
    return cont, categ, target
