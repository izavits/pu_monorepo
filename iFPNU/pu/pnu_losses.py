#
# pnu_losses.py
# Loss functions for PNU risk estimators
#
import torch
from torch.nn import functional as F
import torch.nn as nn


class SigmoidNPU(nn.Module):
    def __init__(self, prior, gamma, beta, eta):
        super(SigmoidNPU, self).__init__()
        if not 0 < prior < 1:
            raise NotImplementedError('The class prior should be in (0, 1)')
        if not 0 < eta <= 1:
            raise NotImplementedError('Eta should be in (0, 1] for PNU risk')
        self.prior = prior
        self.beta = beta
        self.gamma = gamma
        self.eta = eta
        self.positive = 1
        self.negative = 0
        self.unlabeled = -1

    def surrogate_loss(self, inputs, targets):
        inputs = inputs.float()
        targets = targets.float()
        loss = torch.sigmoid(-inputs)
        return loss

    def forward(self, inputs, targets):
        inputs = inputs.float()
        targets = targets.float()
        # Isolate positive and unlabeled
        positive = (targets == self.positive).float()
        negative = (targets == self.negative).float()
        unlabeled = (targets == self.unlabeled).float()
        # Get number of positives and unlabeled
        n_positive = max(1., positive.sum().item())
        n_negative = max(1., negative.sum().item())
        n_unlabeled = max(1., unlabeled.sum().item())
        # Pass through loss function
        y_positive = self.surrogate_loss(inputs, targets)
        y_negative = self.surrogate_loss(-inputs, targets)
        y_unlabeled = self.surrogate_loss(inputs, targets)
        # Calculate partial risks
        positive_risk = torch.sum(self.prior * positive / n_positive * y_positive)
        negative_risk = torch.sum((1-self.prior) * negative / n_negative * y_negative)
        unlabeled_risk = torch.sum(
            (unlabeled / n_unlabeled - self.prior * positive / n_positive) * y_unlabeled)
        # Perform gradient ascent to avoid overfitting (nnPU)
        if unlabeled_risk.item() < -self.beta:
            objective = (positive_risk - self.beta + self.gamma * unlabeled_risk).detach() - \
                        self.gamma * unlabeled_risk
        # Else perform standard (uPU) learning
        else:
            objective = positive_risk + unlabeled_risk
        pnu_risk = (1-self.eta)*(positive_risk+negative_risk) + self.eta*objective
        return pnu_risk


class FocalLossNPU(nn.Module):
    def __init__(self, prior, alpha, gamma, beta, eta):
        super(FocalLossNPU, self).__init__()
        if not 0 < prior < 1:
            raise NotImplementedError('The class prior should be in (0, 1)')
        self.prior = prior
        self.gamma = gamma
        self.beta = beta
        self.alpha = alpha
        self.eta = eta
        self.positive = 1
        self.negative = 0
        self.unlabeled = -1

    def surrogate_loss(self, inputs, targets):
        inputs = inputs.float()
        targets = targets.float()
        p = torch.sigmoid(inputs)
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** self.gamma)
        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss
        return loss

    def forward(self, inputs, targets):
        inputs = inputs.float()
        targets = targets.float()
        # Isolate positive and unlabeled
        positive = (targets == self.positive).float()
        negative = (targets == self.negative).float()
        unlabeled = (targets == self.unlabeled).float()
        # Get number of positives and unlabeled
        n_positive = max(1., positive.sum().item())
        n_negative = max(1., negative.sum().item())
        n_unlabeled = max(1., unlabeled.sum().item())
        # Pass through loss function
        y_positive = self.surrogate_loss(inputs, targets)
        y_negative = self.surrogate_loss(inputs, targets)
        y_unlabeled = self.surrogate_loss(inputs, targets)
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
            objective = positive_risk + unlabeled_risk.item()
        pnu_risk = (1 - self.eta) * (positive_risk + negative_risk) + self.eta * objective
        return pnu_risk
