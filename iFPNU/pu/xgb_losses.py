#
# xgb_losses.py
# Loss functions for gradient tree boosting
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


class ResidualsCalculator(nn.Module):
    def __init__(self, loss_func, predicted_values, prior, alpha=-1, gamma=1, beta=0, reduce="none"):
        super(ResidualsCalculator, self).__init__()
        self.loss_func = loss_func
        self.prior = prior
        self.alpha = alpha
        self.gamma = gamma
        self.beta = beta
        self.reduce = reduce
        self.positive = 1
        self.unlabeled = 0
        self.predicted_values = nn.Parameter(data=torch.zeros(predicted_values.shape), requires_grad=True)
        self.predicted_values.data = predicted_values

    def forward(self):
        my_parameters = self.predicted_values
        return my_parameters

    def loss(self, targets):
        if self.loss_func == 'focal_loss':
            return self.focal_loss(targets)
        elif self.loss_func == 'focal_loss_nnpu':
            return self.focal_loss_nnpu(targets)
        elif self.loss_func == 'sample_dependent_focal_loss_nnpu':
            return self.sample_dependent_focal_loss_nnpu(targets)
        elif self.loss_func == 'sigmoid_nnpu':
            return self.sigmoid_loss(targets)
        else:
            raise NotImplementedError('Loss criterion not implemented.')

    def focal_loss(self, targets):
        logodds = self.forward()
        p = 1.0 / (1.0 + torch.exp(-logodds))
        ce_loss = F.binary_cross_entropy_with_logits(logodds, targets, reduction="none")
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

    def surrogate_fl(self, inputs, targets):
        #p = 1.0 / (1.0 + torch.exp(inputs))
        p = torch.sigmoid(inputs)
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** self.gamma)
        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss
        return loss

    def focal_loss_nnpu(self, targets):
        logodds = self.forward()
        # Isolate positive and unlabeled
        positive = (targets == self.positive).float()
        unlabeled = (targets == self.unlabeled).float()
        # Get number of positives and unlabeled
        n_positive = max(1., positive.sum().item())
        n_unlabeled = max(1., unlabeled.sum().item())
        # Pass through loss function
        y_positive = self.surrogate_fl(logodds, targets)
        y_unlabeled = self.surrogate_fl(-logodds, targets)
        # Calculate partial risks
        positive_risk = self.prior * torch.sum((positive * y_positive) / n_positive)
        ####negative_risk = torch.sum(
        #    (unlabeled / n_unlabeled - self.prior * positive / n_positive) * y_unlabeled)
        negative_risk = -self.prior * torch.sum((positive * y_unlabeled) / n_positive) + torch.sum(
            unlabeled / n_unlabeled)

        # Oversampled risks
        #prior_prime = 0.5
        #positive_risk = prior_prime * torch.sum((positive * y_positive) / n_positive)
        #negative_risk = (1 - prior_prime) / (n_unlabeled * (1 - self.prior)) * torch.sum(positive * y_unlabeled) - (
        #            (1 - prior_prime) * self.prior / (n_positive * (1 - self.prior))) * torch.sum(unlabeled / n_unlabeled)

        # Perform gradient ascent to avoid overfitting (nnPU)
        if negative_risk.item() < -self.beta:
            objective = -negative_risk
            #objective = (positive_risk - self.beta + 1.0 * negative_risk).detach() - \
            #            1.0 * negative_risk
        # Else perform standard (uPU) learning
        else:
            objective = positive_risk + negative_risk
        return objective

    def get_gamma_list(self, pt):
        gamma_dict = {0.2: 3.0, 0.5: 1.0}
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
        return torch.tensor(gamma_list)

    def surrogate_sample_dependent_fl(self, inputs, targets):
        probs = torch.sigmoid(inputs)
        probs = torch.clamp(probs, 1e-8, 1 - 1e-8)
        gamma = self.get_gamma_list(probs)
        loss_pos = - (1 - probs) ** gamma * torch.log(probs)  # when target == 1
        loss_neg = - probs ** gamma * torch.log(1 - probs)  # when target == 0
        loss = targets * loss_pos + (1 - targets) * loss_neg
        if self.alpha > 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss
        if self.reduce == 'mean':
            return loss.mean()
        elif self.reduce == 'sum':
            return loss.sum()
        else:
            return loss.mean()

        # p = torch.sigmoid(inputs)
        # ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        # p_t = p * targets + (1 - p) * (1 - targets)
        # gamma = self.get_gamma_list(p_t)
        # loss = ce_loss * ((1 - p_t) ** gamma)
        # if self.alpha > 0:
        #     alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        #     loss = alpha_t * loss
        # if self.reduce == 'mean':
        #     return loss.mean()
        # elif self.reduce == 'sum':
        #     return loss.sum()
        # else:
        #     return loss

    def sample_dependent_focal_loss_nnpu(self, targets):
        logodds = self.forward()
        # Isolate positive and unlabeled
        positive = (targets == self.positive).float()
        unlabeled = (targets == self.unlabeled).float()
        # Get number of positives and unlabeled
        n_positive = max(1., positive.sum().item())
        n_unlabeled = max(1., unlabeled.sum().item())
        # Pass through loss function
        y_positive = self.surrogate_sample_dependent_fl(logodds, positive)
        y_unlabeled = self.surrogate_sample_dependent_fl(logodds, unlabeled)
        # Calculate partial risks
        positive_risk = torch.sum(self.prior * positive / n_positive * y_positive)
        #positive_risk = self.prior * torch.sum((positive * y_positive) / n_positive)
        negative_risk = torch.sum((unlabeled / n_unlabeled - self.prior * positive / n_positive) * y_unlabeled)
        #negative_risk = -self.prior * torch.sum((positive * y_unlabeled) / n_positive) + torch.sum(
        #    unlabeled / n_unlabeled)
        # Perform gradient ascent to avoid overfitting (nnPU)
        if negative_risk.item() < -self.beta:
            objective = (positive_risk - self.beta + 1 * negative_risk).detach() - 1.0 * negative_risk
        # Else perform standard (uPU) learning
        else:
            objective = positive_risk + negative_risk
        return 1-objective

    def surrogate_sigmoid(self, inputs, targets):
        inputs = inputs.float()
        loss = torch.sigmoid(-inputs)
        return loss

    def sigmoid_loss(self, targets):
        logodds = self.forward()
        # Isolate positive and unlabeled
        positive = (targets == self.positive).float()
        unlabeled = (targets == self.unlabeled).float()
        # Get number of positives and unlabeled
        n_positive = max(1., positive.sum().item())
        n_unlabeled = max(1., unlabeled.sum().item())
        # Pass through loss function
        y_positive = self.surrogate_sigmoid(logodds, targets)
        y_unlabeled = self.surrogate_sigmoid(-logodds, targets)
        # Calculate partial risks
        positive_risk = torch.sum(self.prior * positive / n_positive * y_positive)
        negative_risk = torch.sum(
            (unlabeled / n_unlabeled - self.prior * positive / n_positive) * y_unlabeled)
        # Perform gradient ascent to avoid overfitting (nnPU)
        if negative_risk.item() < -self.beta:
            objective = -negative_risk
            #objective = (positive_risk - self.beta + 1.0 * negative_risk).detach() - \
            #            1.0 * negative_risk
        # Else perform standard (uPU) learning
        else:
            objective = positive_risk + negative_risk
        return objective


class RiskMinimizerXGB(nn.Module):
    def __init__(self, loss_func, prior, alpha=-1, gamma=1, beta=0, reduce="none"):
        super(RiskMinimizerXGB, self).__init__()
        self.loss_func = loss_func
        self.prior = prior
        self.alpha = alpha
        self.gamma = gamma
        self.beta = beta
        self.reduce = reduce
        self.positive = 1
        self.unlabeled = 0
        self.current_leaf_value = nn.Parameter(data=FloatTensor([0.0]), requires_grad=True)

    def reinitialize_variable(self):
        self.current_leaf_value.data = FloatTensor([0.0])

    def refine_previous_predictions(self, previous_predictions):
        new_predictions = previous_predictions + self.current_leaf_value
        return new_predictions

    def loss(self, previous_predictions, targets_leaf_tensor):
        if self.loss_func == 'focal_loss':
            return self.focal_loss(previous_predictions, targets_leaf_tensor)
        elif self.loss_func == 'focal_loss_nnpu':
            return self.focal_loss_nnpu(previous_predictions, targets_leaf_tensor)
        elif self.loss_func == 'sample_dependent_focal_loss_nnpu':
            return self.sample_dependent_focal_loss_nnpu(previous_predictions, targets_leaf_tensor)
        elif self.loss_func == 'sigmoid_nnpu':
            return self.sigmoid_loss(previous_predictions, targets_leaf_tensor)
        else:
            raise NotImplementedError('Loss criterion not implemented.')

    def focal_loss(self, previous_predictions, targets_leaf_tensor):
        logodds = self.refine_previous_predictions(previous_predictions)
        p = 1.0 / (1.0 + torch.exp(-logodds))
        ce_loss = F.binary_cross_entropy_with_logits(logodds, targets_leaf_tensor, reduction="none")
        p_t = p * targets_leaf_tensor + (1 - p) * (1 - targets_leaf_tensor)
        loss = ce_loss * ((1 - p_t) ** self.gamma)
        if self.alpha >= 0:
            alpha_t = self.alpha * targets_leaf_tensor + (1 - self.alpha) * (1 - targets_leaf_tensor)
            loss = alpha_t * loss
        if self.reduce == "mean":
            loss = loss.mean()
        elif self.reduce == "sum":
            loss = loss.sum()
        return loss

    def surrogate_fl(self, inputs, targets):
        #p = 1.0 / (1.0 + torch.exp(inputs))
        p = torch.sigmoid(inputs)
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** self.gamma)
        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss
        return loss

    def surrogate_pos_loss_fl(self, inputs, targets):
        inputs = inputs.float()
        targets = targets.float()
        p = torch.sigmoid(inputs)
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        p_t = p*targets
        alpha_t = -self.alpha * targets
        loss = alpha_t * ce_loss * ((1 - p_t) ** self.gamma)
        return loss

    def surrogate_neg_loss_fl(self, inputs, targets):
        inputs = inputs.float()
        targets = targets.float()
        p = torch.sigmoid(inputs)
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        p_t = (1 - p) * (1 - targets)
        alpha_t = -(1 - self.alpha) * (1 - targets)
        loss = alpha_t * ce_loss * ((p_t) ** self.gamma)
        return loss

    def focal_loss_nnpu(self, previous_predictions, targets_leaf_tensor):
        logodds = self.refine_previous_predictions(previous_predictions)
        # Isolate positive and unlabeled
        positive = (targets_leaf_tensor == self.positive).float()
        unlabeled = (targets_leaf_tensor == self.unlabeled).float()
        # Get number of positives and unlabeled
        n_positive = max(1., positive.sum().item())
        n_unlabeled = max(1., unlabeled.sum().item())
        # Pass through loss function
        #y_positive = self.surrogate_fl(logodds, targets_leaf_tensor)
        #y_unlabeled = self.surrogate_fl(-logodds, targets_leaf_tensor)

        y_positive = self.surrogate_pos_loss_fl(logodds, targets_leaf_tensor)
        y_unlabeled = self.surrogate_neg_loss_fl(-logodds, targets_leaf_tensor)
        positive_risk = self.prior * torch.sum((positive * y_positive) / n_positive)
        ####negative_risk = torch.sum(
        #    (unlabeled / n_unlabeled - self.prior * positive / n_positive) * y_unlabeled)
        negative_risk = -self.prior * torch.sum((positive * y_unlabeled) / n_positive) + torch.sum(unlabeled / n_unlabeled)

        # Oversampled risks
        #prior_prime = 0.5
        #positive_risk = prior_prime * torch.sum((positive * y_positive) / n_positive)
        #negative_risk = (1 - prior_prime) / (n_unlabeled * (1 - self.prior)) * torch.sum(
        #    positive * y_unlabeled) - (
        #                        (1 - prior_prime) * self.prior / (n_positive * (1 - self.prior))) * torch.sum(
        #    unlabeled / n_unlabeled)

        # Perform gradient ascent to avoid overfitting (nnPU)
        if negative_risk.item() < -self.beta:
            #objective = -negative_risk
            objective = (positive_risk - self.beta + self.gamma * negative_risk).detach() - self.gamma * negative_risk
            #objective = (positive_risk - self.beta + 1.0 * negative_risk).detach() - \
            #            1.0 * negative_risk
        # Else perform standard (uPU) learning
        else:
            objective = positive_risk + negative_risk
        return objective

    def get_gamma_list(self, pt):
        gamma_dict = {0.2: 3.0, 0.5: 1.0}
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
        return torch.tensor(gamma_list)

    def surrogate_sample_dependent_fl(self, inputs, targets):
        probs = torch.sigmoid(inputs)
        probs = torch.clamp(probs, 1e-8, 1 - 1e-8)
        gamma = self.get_gamma_list(probs)
        loss_pos = - (1 - probs) ** gamma * torch.log(probs)  # when target == 1
        loss_neg = - probs ** gamma * torch.log(1 - probs)  # when target == 0
        loss = targets * loss_pos + (1 - targets) * loss_neg
        if self.alpha > 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss
        if self.reduce == 'mean':
            return loss.mean()
        elif self.reduce == 'sum':
            return loss.sum()
        else:
            return loss.mean()

        # p = torch.sigmoid(inputs)
        # ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        # p_t = p * targets + (1 - p) * (1 - targets)
        # gamma = self.get_gamma_list(p_t)
        # loss = ce_loss * ((1 - p_t) ** gamma)
        # if self.alpha > 0:
        #     alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        #     loss = alpha_t * loss
        # if self.reduce == 'mean':
        #     return loss.mean()
        # elif self.reduce == 'sum':
        #     return loss.sum()
        # else:
        #     return loss

    def sample_dependent_focal_loss_nnpu(self, previous_predictions, targets_leaf_tensor):
        logodds = self.refine_previous_predictions(previous_predictions)
        # Isolate positive and unlabeled
        positive = (targets_leaf_tensor == self.positive).float()
        unlabeled = (targets_leaf_tensor == self.unlabeled).float()
        # Get number of positives and unlabeled
        n_positive = max(1., positive.sum().item())
        n_unlabeled = max(1., unlabeled.sum().item())
        # Pass through loss function
        y_positive = self.surrogate_sample_dependent_fl(logodds, positive)
        y_unlabeled = self.surrogate_sample_dependent_fl(logodds, unlabeled)
        # Calculate partial risks
        #positive_risk = self.prior * torch.sum((positive * y_positive) / n_positive)
        #negative_risk = -self.prior * torch.sum((positive * y_unlabeled) / n_positive) + torch.sum(
        #    unlabeled / n_unlabeled)
        positive_risk = torch.sum(self.prior * positive / n_positive * y_positive)
        negative_risk = torch.sum((unlabeled / n_unlabeled - self.prior * positive / n_positive) * y_unlabeled)
        # Perform gradient ascent to avoid overfitting (nnPU)
        if negative_risk.item() < -self.beta:
            objective = (positive_risk - self.beta + 1 * negative_risk).detach() - 1 * negative_risk
        # Else perform standard (uPU) learning
        else:
            objective = positive_risk + negative_risk
        return 1-objective

    def surrogate_sigmoid(self, inputs, targets):
        inputs = inputs.float()
        loss = torch.sigmoid(-inputs)
        return loss

    def sigmoid_loss(self, previous_predictions, targets_leaf_tensor):
        logodds = self.refine_previous_predictions(previous_predictions)
        # Isolate positive and unlabeled
        positive = (targets_leaf_tensor == self.positive).float()
        unlabeled = (targets_leaf_tensor == self.unlabeled).float()
        # Get number of positives and unlabeled
        n_positive = max(1., positive.sum().item())
        n_unlabeled = max(1., unlabeled.sum().item())
        # Pass through loss function
        y_positive = self.surrogate_sigmoid(logodds, targets_leaf_tensor)
        y_unlabeled = self.surrogate_sigmoid(-logodds, targets_leaf_tensor)
        # Calculate partial risks
        positive_risk = torch.sum(self.prior * positive / n_positive * y_positive)
        negative_risk = torch.sum(
            (unlabeled / n_unlabeled - self.prior * positive / n_positive) * y_unlabeled)
        # Perform gradient ascent to avoid overfitting (nnPU)
        if negative_risk.item() < -self.beta:
            objective = -negative_risk
            #objective = (positive_risk - self.beta + 1.0 * negative_risk).detach() - \
            #            1.0 * negative_risk
        # Else perform standard (uPU) learning
        else:
            objective = positive_risk + negative_risk
        return objective
