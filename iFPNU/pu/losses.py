#
# losses.py
# Loss functions for PU risk estimators
#

import torch
from torch.nn import functional as F
import torch.nn as nn


class FocalLoss(nn.Module):
    def __init__(self, alpha=-1, gamma=2, reduce="none"):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduce = reduce

    def forward(self, inputs, targets):
        inputs = inputs.float()
        targets = targets.float()
        p = torch.sigmoid(inputs)
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** self.gamma)
        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss
        if self.reduce == "mean":
            loss = loss.mean()
        elif self.reduce == "sum":
            loss = loss.sum()
        return loss


class FocalLossNNPU(nn.Module):
    def __init__(self, prior, alpha, gamma, beta):
        super(FocalLossNNPU, self).__init__()
        if not 0 < prior < 1:
            raise NotImplementedError('The class prior should be in (0, 1)')
        self.prior = prior
        self.gamma = gamma
        self.beta = beta
        self.alpha = alpha
        self.positive = 1
        self.unlabeled = 0

    def surrogate_pos_loss(self, inputs, targets):
        inputs = inputs.float()
        targets = targets.float()
        p = torch.sigmoid(inputs)
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        p_t = p*targets
        alpha_t = -self.alpha * targets
        loss = alpha_t * ce_loss * ((1 - p_t) ** self.gamma)
        return loss

    def surrogate_neg_loss(self, inputs, targets):
        inputs = inputs.float()
        targets = targets.float()
        p = torch.sigmoid(inputs)
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        p_t = (1 - p) * (1 - targets)
        alpha_t = -(1 - self.alpha) * (1 - targets)
        loss = alpha_t * ce_loss * ((p_t) ** self.gamma)
        return loss

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
        unlabeled = (targets == self.unlabeled).float()
        # Get number of positives and unlabeled
        n_positive = max(1., positive.sum().item())
        n_unlabeled = max(1., unlabeled.sum().item())
        # Pass through loss function
        #y_positive = self.surrogate_loss(inputs, targets)
        #y_unlabeled = self.surrogate_loss(-inputs, targets)
        y_positive = self.surrogate_pos_loss(inputs, targets)
        y_unlabeled = self.surrogate_neg_loss(-inputs, targets)

        # Calculate partial risks
        positive_risk = torch.sum(self.prior * positive / n_positive * y_positive)
        negative_risk = torch.sum(
            (unlabeled / n_unlabeled - self.prior * positive / n_positive) * y_unlabeled)
        # Perform gradient ascent to avoid overfitting (nnPU)
        if negative_risk.item() < -self.beta:
            objective = - 1.0*negative_risk
           #objective = (positive_risk - self.beta + 1.0 * negative_risk).detach() - \
           #             1.0 * negative_risk
        # Else perform standard (uPU) learning
        else:
            objective = positive_risk + negative_risk
        return objective


class SampleDependentFocalLoss(nn.Module):
    def __init__(self, alpha=-1, gamma=3, reduce="none", device=None):
        super(SampleDependentFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduce = reduce
        self.device = device

    def get_gamma_list(self, pt):
        gamma_dict = {0.2: 5.0, 0.5: 3.0}
        gamma_list = []
        batch_size = pt.shape[0]
        for i in range(batch_size):
            pt_sample = pt[i].item()
            if pt_sample >= 0.5:
                gamma_list.append(self.gamma)
                continue
            # Choosing the gamma for the sample
            for key in sorted(gamma_dict.keys()):
                if pt_sample < key:
                    gamma_list.append(gamma_dict[key])
                    break
        return torch.tensor(gamma_list).to(self.device)

    def forward(self, inputs, targets):
        inputs = inputs.float()
        targets = targets.float()
        p = torch.sigmoid(inputs)
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        p_t = p * targets + (1 - p) * (1 - targets)
        gamma = self.get_gamma_list(p_t)
        loss = ce_loss * ((1 - p_t) ** gamma)
        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss
        if self.reduce == "mean":
            loss = loss.mean()
        elif self.reduce == "sum":
            loss = loss.sum()
        return loss


class SampleDependentFocalLossNNPU(nn.Module):
    def __init__(self, prior, alpha, gamma, beta, device=None):
        super(SampleDependentFocalLossNNPU, self).__init__()
        if not 0 < prior < 1:
            raise NotImplementedError('The class prior should be in (0, 1)')
        self.prior = prior
        self.beta = beta
        self.alpha = alpha
        self.gamma = gamma
        self.positive = 1
        self.unlabeled = 0
        self.device = device

    def get_gamma_list(self, pt):
        gamma_dict = {0.1: 1.0, 0.2: 1.0}
        gamma_list = []
        batch_size = pt.shape[0]
        for i in range(batch_size):
            pt_sample = pt[i].item()
            if pt_sample >= 0.2:
                gamma_list.append(self.gamma)
                continue
            # Choosing the gamma for the sample
            for key in sorted(gamma_dict.keys()):
                if pt_sample < key:
                    gamma_list.append(gamma_dict[key])
                    break
        return torch.tensor(gamma_list).to(self.device)

    def surrogate_loss(self, inputs, targets):
        inputs = inputs.float()
        targets = targets.float()
        p = torch.sigmoid(inputs)
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        p_t = p * targets + (1 - p) * (1 - targets)
        gamma = self.get_gamma_list(p_t)
        loss = ce_loss * ((1 - p_t) ** gamma)
        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss
        return loss

    def forward(self, inputs, targets):
        inputs = inputs.float()
        targets = targets.float()
        # Isolate positive and unlabeled
        positive = (targets == self.positive).float()
        unlabeled = (targets == self.unlabeled).float()
        # Get number of positives and unlabeled
        n_positive = max(1., positive.sum().item())
        n_unlabeled = max(1., unlabeled.sum().item())
        # Pass through loss function
        y_positive = self.surrogate_loss(inputs, targets)
        y_unlabeled = self.surrogate_loss(-inputs, targets)
        # Calculate partial risks
        positive_risk = torch.sum(self.prior * positive / n_positive * y_positive)
        negative_risk = torch.sum(
            (unlabeled / n_unlabeled - self.prior * positive / n_positive) * y_unlabeled)
        # Perform gradient ascent to avoid overfitting (nnPU)
        if negative_risk.item() < -self.beta:
            objective = (positive_risk - self.beta + self.gamma * negative_risk).detach() - \
                        self.gamma * negative_risk
        # Else perform standard (uPU) learning
        else:
            objective = positive_risk + negative_risk
        return objective


class SigmoidNNPU(nn.Module):
    def __init__(self, prior, gamma, beta):
        super(SigmoidNNPU, self).__init__()
        if not 0 < prior < 1:
            raise NotImplementedError('The class prior should be in (0, 1)')
        self.prior = prior
        self.beta = beta
        self.gamma = gamma
        self.positive = 1
        self.unlabeled = 0

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
        unlabeled = (targets == self.unlabeled).float()
        # Get number of positives and unlabeled
        n_positive = max(1., positive.sum().item())
        n_unlabeled = max(1., unlabeled.sum().item())
        # Pass through loss function
        y_positive = self.surrogate_loss(inputs, targets)
        y_unlabeled = self.surrogate_loss(-inputs, targets)
        # Calculate partial risks
        positive_risk = torch.sum(self.prior * positive / n_positive * y_positive)
        negative_risk = torch.sum(
            (unlabeled / n_unlabeled - self.prior * positive / n_positive) * y_unlabeled)
        # Perform gradient ascent to avoid overfitting (nnPU)
        if negative_risk.item() < -self.beta:
            objective = -negative_risk
            #objective = (positive_risk - self.beta + self.gamma * negative_risk).detach() - \
            #            self.gamma * negative_risk
        # Else perform standard (uPU) learning
        else:
            objective = positive_risk + negative_risk
        return objective

