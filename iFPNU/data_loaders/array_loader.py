#
# array_loader.py
# Dataset loader for np arrays
#

from torch.utils.data import Dataset, Subset
import numpy as np


class ArrayDataset(Dataset):
    def __init__(self, x_train, y_train, x_test, y_test):
        """
        Load the data as np arrays
        :param x_train: the training data
        :param y_train: y train classes
        :param x_test: the test data
        :param y_test: the test classes
        """
        self.X_train = x_train
        self.y_train = y_train
        self.X_test = x_test
        self.y_test = y_test
        self.X_train = self.X_train.astype('float32')
        self.y_train = self.y_train.astype('float32')
        self.X_test = self.X_test.astype('float32')
        self.y_test = self.y_test.astype('float32')
        self.X = np.concatenate((self.X_train, self.X_test), axis=0)
        self.y = np.concatenate((self.y_train, self.y_test), axis=None)
        self.y = self.y.reshape((len(self.y), 1))

    def __getitem__(self, idx):
        """
        Get a row at a specific index
        :param idx: the specified index
        :return: the row
        """
        return [self.X[idx], self.y[idx]]

    def get_splits(self):
        """
        Get indexes for train and test rows
        :return: the split
        """
        train_idx = list(range(len(self.X_train)))
        test_idx = list(range(len(self.X_train), len(self.X_train) + len(self.X_test)))
        # Subset dataset for train and val
        train_dataset = Subset(self, train_idx)
        test_dataset = Subset(self, test_idx)
        return train_dataset, test_dataset