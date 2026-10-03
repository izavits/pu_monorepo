#
# gradient_boost.py
# Pytorch based Gradient Tree Boosting
#

from sklearn.tree import DecisionTreeRegressor
from pu.xgb_losses import RiskMinimizerXGB, ResidualsCalculator
from pu.xgb_pnu_losses import RiskMinimizerXGBPNU, ResidualsCalculatorPNU
import torch
import numpy as np

if torch.cuda.is_available():
    FloatTensor = torch.cuda.FloatTensor
    LongTensor = torch.cuda.LongTensor
    ByteTensor = torch.cuda.ByteTensor
else:
    FloatTensor = torch.FloatTensor
    LongTensor = torch.LongTensor
    ByteTensor = torch.ByteTensor


def fit_regression_tree_classifier_to_residuals(X_data, y_data, max_depth):  # y_data -> residuals
    tree_regressor = DecisionTreeRegressor(max_depth=max_depth)
    tree_regressor.fit(X_data, y_data)
    leaf_buckets = []
    for i in range(X_data.shape[0]):
        leaf_buckets.append(tuple(tree_regressor.decision_path(X_data[i, :].reshape(1, -1)).todok().keys()))
    unique_paths = list(set(leaf_buckets))
    return leaf_buckets, unique_paths, tree_regressor


class PytorchBasedGenericGradientBoost():
    def __init__(self, n_trees, max_depth, loss_func, GRADIENT_BOOST_LEARNING_RATE=0.1,
                 MINIMIZER_LEARNING_RATE=0.001, MINIMIZER_TRAINING_EPOCHS=5000, prior=0.1, alpha=-1, gamma=2,
                 beta=0, eta=0.5, reduce="mean", pnu=False):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.loss_func = loss_func
        self.gradient_boost_learning_rate = GRADIENT_BOOST_LEARNING_RATE
        self.minimizer_learning_rate = MINIMIZER_LEARNING_RATE
        self.minimizer_training_epochs = MINIMIZER_TRAINING_EPOCHS
        self.prior = prior
        self.alpha = alpha
        self.gamma = gamma
        self.beta = beta
        self.eta = eta
        self.reduce = reduce
        # Variables to hold output of algorithm
        self.initial_prediction = None
        self.regression_trees = []
        # Get an instance of a minimizer
        self.pnu = pnu
        if self.pnu:
            self.minimizer = RiskMinimizerXGBPNU(self.loss_func, self.prior, self.alpha, self.gamma,
                                                 self.beta, self.eta, self.reduce)
        else:
            self.minimizer = RiskMinimizerXGB(self.loss_func, self.prior, self.alpha, self.gamma, self.beta,
                                              self.reduce)
        self.minimizer_optimizer = torch.optim.Adam(self.minimizer.parameters(),
                                                    lr=self.minimizer_learning_rate)

    def minimize_loss_function(self, targets, previous_predictions):
        self.minimizer.reinitialize_variable()
        for training_epoch in range(self.minimizer_training_epochs):
            targets_leaf_tensor = FloatTensor(targets)
            loss = self.minimizer.loss(previous_predictions, targets_leaf_tensor)
            self.minimizer.zero_grad()
            loss.backward()
            self.minimizer_optimizer.step()
        return [el for el in self.minimizer.parameters()][0].cpu().detach().numpy()[0]

    def compute_residuals(self, targets, predicted_values):
        if self.pnu:
            model = ResidualsCalculatorPNU(self.loss_func, predicted_values, self.prior, self.alpha, self.gamma,
                                        self.beta, self.eta, self.reduce)
        else:
            model = ResidualsCalculator(self.loss_func, predicted_values, self.prior, self.alpha, self.gamma,
                                    self.beta, self.reduce)
        loss = model.loss(targets)
        model.zero_grad()
        loss.backward()
        residuals = model.predicted_values.grad.clone()  # deep copy of the input/gradients
        return residuals

    def fit(self, X, y):
        X_values = X.copy()
        y_values = y.copy()
        # Initialization phase
        initial_values = torch.zeros(y_values.shape)
        self.initial_prediction = self.minimize_loss_function(y_values, initial_values)
        prediction_values = np.ones(y_values.shape) * self.initial_prediction

        for classifier_index in range(self.n_trees):
            self.regression_trees.append({"tree_index": classifier_index})
            residuals = self.compute_residuals(FloatTensor(y_values), FloatTensor(prediction_values))
            leaf_buckets, unique_clusters, tree_regressor = fit_regression_tree_classifier_to_residuals(
                X_values, residuals.cpu(), self.max_depth)
            self.regression_trees[-1]["tree_regressor"] = tree_regressor

            X_values_temp = np.array([])
            y_values_temp = np.array([])
            prediction_values_temp = np.array([])

            for unique_cluster in unique_clusters:
                indices = [1 if el == unique_cluster else 0 for el in leaf_buckets]
                y_leaf = y_values[np.array(indices) == 1]
                X_leaf = X_values[np.array(indices) == 1]
                predictions_leaf = prediction_values[np.array(indices) == 1]
                prediction_for_leaf = self.minimize_loss_function(FloatTensor(np.array(y_leaf)),
                                                                  FloatTensor(predictions_leaf))
                predictions_for_leaf_array = np.ones(
                    y_leaf.shape) * self.gradient_boost_learning_rate * prediction_for_leaf + predictions_leaf
                self.regression_trees[-1][str(unique_cluster)] = prediction_for_leaf
                X_values_temp = X_leaf if X_values_temp.shape == (0,) else np.append(X_values_temp, X_leaf,
                                                                                     axis=0)
                y_values_temp = np.append(y_values_temp, y_leaf)
                prediction_values_temp = np.append(prediction_values_temp, predictions_for_leaf_array)
            y_values = y_values_temp
            X_values = X_values_temp
            prediction_values = prediction_values_temp

    def predict(self, X):
        predict_proba = []
        for index in range(X.shape[0]):
            prediction = self.initial_prediction
            for tree_index in range(self.n_trees):
                tree = self.regression_trees[tree_index]
                prediction += self.gradient_boost_learning_rate * tree[str(tuple(
                    tree["tree_regressor"].decision_path(X[index, :].reshape(1, -1)).todok().keys()))]
            predict_proba.append(prediction)
        predictions = torch.sigmoid(torch.tensor(predict_proba)).numpy()
        return predictions
