#
# xgb_pnu_losses.py
# Loss functions for grafient tree boosting
#

import torch
from torch.nn import functional as F
import torch.nn as nn

if torch.cuda.is_available():
    FloatTensor = torch.cuda.FloatTensor
    LongTensor = torch.cuda.LongTensor
    ByteTensor = torch.cuda.ByteTensor
else:
    FloatTensor = torch.FloatTensor
    LongTensor = torch.LongTensor
    ByteTensor = torch.ByteTensor


class ResidualsCalculatorPNU(nn.Module):
    def __init__(self, loss_func, predicted_values, prior, alpha=-1, gamma=2, beta=0, eta=0.5, reduce="none"):
        super(ResidualsCalculatorPNU, self).__init__()
        self.loss_func = loss_func
        self.prior = prior
        self.alpha = alpha
        self.gamma = gamma
        self.beta = beta
        self.eta = eta
        self.reduce = reduce
        self.positive = 1
        self.negative = 0
        self.unlabeled = -1
        self.predicted_values = nn.Parameter(data=torch.zeros(predicted_values.shape), requires_grad=True)
        self.predicted_values.data = predicted_values

    def forward(self):
        my_parameters = self.predicted_values
        return my_parameters

    def loss(self, targets):
        if self.loss_func == 'focal_loss_npu':
            return self.focal_loss_mpu(targets)
        elif self.loss_func == 'sigmoid_npu':
            return self.sigmoid_npu(targets)
        else:
            raise NotImplementedError('Loss criterion not implemented.')

    def surrogate_fl(self, inputs, targets):
        p = 1.0 / (1.0 + torch.exp(-inputs))
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** self.gamma)
        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss
        return loss

    def focal_loss_mpu(self, targets):
        logodds = self.forward()
        # Isolate positive, negative and unlabeled
        positive = (targets == self.positive).float()
        negative = (targets == self.negative).float()
        unlabeled = (targets == self.unlabeled).float()
        # Get number of positives, negatives and unlabeled
        n_positive = max(1., positive.sum().item())
        n_negative = max(1., negative.sum().item())
        n_unlabeled = max(1., unlabeled.sum().item())
        # Pass through loss function
        y_positive = self.surrogate_fl(logodds, positive)
        y_negative = self.surrogate_fl(logodds, negative)
        y_unlabeled = self.surrogate_fl(logodds, unlabeled)
        # Calculate partial risks
        positive_risk = torch.sum(self.prior * positive / n_positive * y_positive)
        negative_risk = torch.sum((1 - self.prior) * negative / n_negative * y_negative)
        unlabeled_risk = torch.sum(
            (unlabeled / n_unlabeled - self.prior * positive / n_positive) * y_unlabeled)
        # Perform gradient ascent to avoid overfitting (nnPU)
        if unlabeled_risk.item() < -self.beta:
            objective = (positive_risk - self.beta + 1.0 * unlabeled_risk).detach() - \
                        1.0 * unlabeled_risk
        # Else perform standard (uPU) learning
        else:
            objective = positive_risk + unlabeled_risk
        pnu_risk = (1 - self.eta) * (positive_risk + negative_risk) + self.eta * objective
        return pnu_risk

    def surrogate_sigmoid(self, inputs, targets):
        inputs = inputs.float()
        loss = torch.sigmoid(-inputs)
        return loss

    def sigmoid_npu(self, targets):
        logodds = self.forward()
        # Isolate positive, negative and unlabeled
        positive = (targets == self.positive).float()
        negative = (targets == self.negative).float()
        unlabeled = (targets == self.unlabeled).float()
        # Get number of positives, negatives and unlabeled
        n_positive = max(1., positive.sum().item())
        n_negative = max(1., negative.sum().item())
        n_unlabeled = max(1., unlabeled.sum().item())
        # Pass through loss function
        y_positive = self.surrogate_sigmoid(logodds, targets)
        y_negative = self.surrogate_sigmoid(-logodds, targets)
        y_unlabeled = self.surrogate_sigmoid(logodds, targets)
        # Calculate partial risks
        positive_risk = torch.sum(self.prior * positive / n_positive * y_positive)
        negative_risk = torch.sum((1 - self.prior) * negative / n_negative * y_negative)
        unlabeled_risk = torch.sum(
            (unlabeled / n_unlabeled - self.prior * positive / n_positive) * y_unlabeled)
        # Perform gradient ascent to avoid overfitting (nnPU)
        if negative_risk.item() < -self.beta:
            objective = (positive_risk - self.beta + self.gamma * unlabeled_risk).detach() - \
                        self.gamma * unlabeled_risk
        # Else perform standard (uPU) learning
        else:
            objective = positive_risk + unlabeled_risk
        pnu_risk = (1 - self.eta) * (positive_risk + negative_risk) + self.eta * objective
        return pnu_risk


class RiskMinimizerXGBPNU(nn.Module):
    def __init__(self, loss_func, prior, alpha=-1, gamma=2, beta=0, eta=0.5, reduce="none"):
        super(RiskMinimizerXGBPNU, self).__init__()
        self.loss_func = loss_func
        self.prior = prior
        self.alpha = alpha
        self.gamma = gamma
        self.beta = beta
        self.eta = eta
        self.reduce = reduce
        self.positive = 1
        self.negative = 0
        self.unlabeled = -1
        self.current_leaf_value = nn.Parameter(data=FloatTensor([0.0]), requires_grad=True)

    def reinitialize_variable(self):
        self.current_leaf_value.data = FloatTensor([0.0])

    def refine_previous_predictions(self, previous_predictions):
        new_predictions = previous_predictions + self.current_leaf_value
        return new_predictions

    def loss(self, previous_predictions, targets_leaf_tensor):
        if self.loss_func == 'focal_loss_npu':
            return self.focal_loss_npu(previous_predictions, targets_leaf_tensor)
        elif self.loss_func == 'sigmoid_npu':
            return self.sigmoid_npu(previous_predictions, targets_leaf_tensor)
        else:
            raise NotImplementedError('Loss criterion not implemented.')

    def surrogate_fl(self, inputs, targets):
        p = 1.0 / (1.0 + torch.exp(-inputs))
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** self.gamma)
        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss
        return loss

    def focal_loss_npu(self, previous_predictions, targets_leaf_tensor):
        logodds = self.refine_previous_predictions(previous_predictions)
        # Isolate positive, negative and unlabeled
        positive = (targets_leaf_tensor == self.positive).float()
        negative = (targets_leaf_tensor == self.negative).float()
        unlabeled = (targets_leaf_tensor == self.unlabeled).float()
        # Get number of positives, negatives and unlabeled
        n_positive = max(1., positive.sum().item())
        n_negative = max(1., negative.sum().item())
        n_unlabeled = max(1., unlabeled.sum().item())
        # Pass through loss function
        y_positive = self.surrogate_fl(logodds, positive)
        y_negative = self.surrogate_fl(logodds, negative)
        y_unlabeled = self.surrogate_fl(logodds, unlabeled)
        # Calculate partial risks
        positive_risk = torch.sum(self.prior * positive / n_positive * y_positive)
        negative_risk = torch.sum((1-self.prior) * negative / n_negative * y_negative)
        unlabeled_risk = torch.sum(
            (unlabeled / n_unlabeled - self.prior * positive / n_positive) * y_unlabeled)
        # Perform gradient ascent to avoid overfitting (nnPU)
        if unlabeled_risk.item() < -self.beta:
            objective = (positive_risk - self.beta + 1.0 * unlabeled_risk).detach() - \
                        1.0 * unlabeled_risk
        # Else perform standard (uPU) learning
        else:
            objective = positive_risk + unlabeled_risk
        pnu_risk = (1 - self.eta) * (positive_risk + negative_risk) + self.eta * objective
        return pnu_risk

    def surrogate_sigmoid(self, inputs, targets):
        inputs = inputs.float()
        loss = torch.sigmoid(-inputs)
        return loss

    def sigmoid_npu(self, previous_predictions, targets_leaf_tensor):
        logodds = self.refine_previous_predictions(previous_predictions)
        # Isolate positive, negative and unlabeled
        positive = (targets_leaf_tensor == self.positive).float()
        negative = (targets_leaf_tensor == self.negative).float()
        unlabeled = (targets_leaf_tensor == self.unlabeled).float()
        # Get number of positives, negatives and unlabeled
        n_positive = max(1., positive.sum().item())
        n_negative = max(1., negative.sum().item())
        n_unlabeled = max(1., unlabeled.sum().item())
        # Pass through loss function
        y_positive = self.surrogate_sigmoid(logodds, targets_leaf_tensor)
        y_negative = self.surrogate_sigmoid(-logodds, targets_leaf_tensor)
        y_unlabeled = self.surrogate_sigmoid(-logodds, targets_leaf_tensor)
        # Calculate partial risks
        positive_risk = torch.sum(self.prior * positive / n_positive * y_positive)
        negative_risk = torch.sum((1 - self.prior) * negative / n_negative * y_negative)
        unlabeled_risk = torch.sum(
            (unlabeled / n_unlabeled - self.prior * positive / n_positive) * y_unlabeled)
        # Perform gradient ascent to avoid overfitting (nnPU)
        if negative_risk.item() < -self.beta:
            objective = (positive_risk - self.beta + self.gamma * unlabeled_risk).detach() - \
                        self.gamma * unlabeled_risk
        # Else perform standard (uPU) learning
        else:
            objective = positive_risk + unlabeled_risk
        pnu_risk = (1 - self.eta) * (positive_risk + negative_risk) + self.eta * objective
        return pnu_risk
