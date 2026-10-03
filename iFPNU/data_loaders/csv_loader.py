#
# csv_loader.py
# Dataset loader for csv files
#
import torch
from pandas import read_csv
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, Subset
import numpy as np


class CSVDataset(Dataset):
    def __init__(self, path):
        """
        Load the dataset csv file as a dataframe.
        Store the inputs and outputs and label encode outputs
        :param path: the dataset path
        """
        df = read_csv(path, header=None)
        self.X = df.values[:, :-1]
        self.y = df.values[:, -1]
        # Ensure input data is floats
        self.X = self.X.astype('float32')
        self.y = self.y.astype('float32')
        self.y = self.y.reshape((len(self.y), 1))

    def __len__(self):
        """
        Get the number of rows in the dataset
        :return: number of rows
        """
        return len(self.X)

    def __getitem__(self, idx):
        """
        Get a row at a specific index
        :param idx: the specified index
        :return: the row
        """
        return [self.X[idx], self.y[idx]]

    def get_splits(self, n_test=0.30, random_state=0):
        """
        Get indexes for train and test rows
        :param n_test: size of test set
        :param random_state: random state
        :return: a random split of the data based on the test set size
        """
        train_idx, test_idx = train_test_split(np.arange(len(self.X)),
                                               test_size=n_test,
                                               random_state=random_state,
                                               shuffle=True,
                                               stratify=self.y)
        # Subset dataset for train and val
        train_dataset = Subset(self, train_idx)
        test_dataset = Subset(self, test_idx)
        return train_dataset, test_dataset

    def update_y(self, y_):
        """
        Update y data
        :param y_: the new y data
        """
        for i in range(len(y_)):
            self.y[i] = y_[i]
