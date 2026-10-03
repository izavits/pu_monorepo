#
# mlp.py
# Multi layer perceptron
#

from torch.nn import Module
from torch.nn import ReLU
from torch.nn import Linear
from torch.nn import Sigmoid
from torch.nn.init import kaiming_uniform_
from torch.nn.init import xavier_uniform_


class MLP(Module):
    def __init__(self, n_inputs):
        """
        Model definition.
        :param n_inputs: input dimension
        """
        super(MLP, self).__init__()
        # Input to first hidden layer
        self.hidden1 = Linear(n_inputs, 100)
        kaiming_uniform_(self.hidden1.weight, nonlinearity='relu')
        self.act1 = ReLU()
        # Second hidden layer
        self.hidden2 = Linear(100, 100)
        kaiming_uniform_(self.hidden2.weight, nonlinearity='relu')
        self.act2 = ReLU()
        # Third hidden layer
        self.hidden3 = Linear(100, 100)
        kaiming_uniform_(self.hidden3.weight, nonlinearity='relu')
        self.act3 = ReLU()
        # Forth hidden layer
        self.hidden4 = Linear(100, 100)
        kaiming_uniform_(self.hidden4.weight, nonlinearity='relu')
        self.act4 = ReLU()
        # Fifth hidden layer and output
        self.hidden5 = Linear(100, 1)
        xavier_uniform_(self.hidden5.weight)
        self.act5 = Sigmoid()

    def forward(self, X):
        """
        Forward propagate input
        :param X: input data
        :return: model output
        """
        # Input to first hidden layer
        X = self.hidden1(X)
        X = self.act1(X)
        # Second hidden layer
        X = self.hidden2(X)
        X = self.act2(X)
        # Third hidden layer
        X = self.hidden3(X)
        X = self.act3(X)
        # Forth hidden layer
        X = self.hidden4(X)
        X = self.act4(X)
        # Fifth hidden layer and output
        X = self.hidden5(X)
        X = self.act5(X)
        return X
