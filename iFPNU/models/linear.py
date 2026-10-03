#
# linear.py
# Linear classifier
#

from torch.nn import Module
from torch.nn import Linear
from torch.nn import Sigmoid


class LinearClassifier(Module):
    def __init__(self, n_imputs):
        """
        Model definition
        :param n_imputs: input dimension
        """
        super(LinearClassifier, self).__init__()
        self.linear = Linear(n_imputs, 1)
        self.act = Sigmoid()

    def forward(self, X):
        """
        Forward propagate input
        :param X: input data
        :return: model output
        """
        X = self.linear(X)
        X = self.act(X)
        return X
