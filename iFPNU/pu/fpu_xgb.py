# Focal loss for PU learning
# Custom loss for XGB classifier

import numpy as np
from scipy.misc import derivative


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class PUFLLossNonNegative:
    def __init__(self, gamma, pi_p, clip_value=1e-7):
        """
        Non-Negative PU Learning Focal Loss for XGBoost.
        Parameters:
        - r1: Focusing parameter of focal loss.
        - pi_p: Estimated class prior P(Y=1).
        - clip_value: Clipping value for numerical stability.
        """
        self.gamma = gamma
        self.pi_p = pi_p  # Class prior P(Y=1)
        self.clip_value = clip_value

    def __call__(self, y_true, preds):
        """
        Computes gradient and hessian for XGBoost.
        Parameters:
        - y_true: Labels (1 for positives, 0 for unlabeled).
        - preds: Raw XGBoost predictions (logits).
        Returns:
        - grad: Gradient of the loss.
        - hess: Hessian (second derivative) of the loss.
        """
        prob = 1 / (1 + np.exp(-preds))  # Convert logits to probabilities
        prob = np.clip(prob, self.clip_value, 1 - self.clip_value)  # Stability

        # Compute gradients
        grad_p = self.grad(prob, self.gamma)  # Gradient for positives
        grad_n = self.grad(1 - prob, self.gamma)  # Gradient for unlabeled

        # Compute Hessians
        hess_p = self.hess1(prob, self.gamma) + self.hess2(prob, self.gamma)  # Hessian for positives
        hess_n = self.hess1(1 - prob, self.gamma) + self.hess2(1 - prob, self.gamma)  # Hessian for unlabeled

        # Compute negative risk term and clip it to be non-negative
        neg_risk = (1 - self.pi_p) * grad_n
        #neg_risk_clipped = np.maximum(neg_risk, 0)  # Ensure non-negative risk
        neg_risk_clipped = neg_risk

        # PU Loss Combination
        grad = y_true * grad_p + (1 - y_true) * (self.pi_p * grad_p - neg_risk_clipped)
        hess = y_true * hess_p + (1 - y_true) * (self.pi_p * hess_p + (1 - self.pi_p) * hess_n)

        # Ensure positive Hessian values for stability
        hess = np.maximum(hess, 1e-6)
        return grad, hess

    def grad(self, p, gamma):
        """Computes the focal loss gradient."""
        return ((1 - p) ** (gamma)) * (gamma * p * np.log(p) + p - 1)

    def hess1(self, p, gamma):
        """Computes the first part of the Hessian."""
        return ((1 - p) ** (gamma)) * (
                -gamma * (gamma - 1) * p ** 2 * np.log(p) - 2 * gamma * p * (p - 1) + (p - 1) ** 2)

    def hess2(self, p, gamma):
        """Computes the second part of the Hessian correction term."""
        return ((1 - p) ** (gamma)) * (1 - 2 * p) * (gamma * p * np.log(p) + p - 1)


class FLLoss:
    def __init__(self, r1, clip_value=1e-7):
        self.r1 = r1
        self.clip_value = clip_value

    def __call__(self, y_true, preds):
        pred = 1 / (1 + np.exp(-preds))
        prob = np.clip(pred, self.clip_value, 1 - self.clip_value)
        grad = y_true * self.grad(prob, self.r1) - (1 - y_true) * self.grad(1 - prob, self.r1)
        hess = y_true * (self.hess1(prob, self.r1) + self.hess2(prob, self.r1)) + \
               (1 - y_true) * (self.hess1(1 - prob, self.r1) + self.hess2(1 - prob, self.r1))
        return grad, hess

    def grad(self, p, r):
        return ((1 - p) ** r) * (r * p * np.log(p) + p - 1)

    def hess1(self, p, r):  # dldp2*dp**2
        return ((1 - p) ** r) * (-r * (r - 1) * p ** 2 * np.log(p) - 2 * r * p * (p - 1) + (p - 1) ** 2)

    def hess2(self, p, r):  # dldp*dp2
        return ((1 - p) ** r) * (1 - 2 * p) * (r * p * np.log(p) + p - 1)


def pnu_focal_loss(y_true, gamma=3.0, eta=0.5, pi_p=0.3, clip=1e-7, lgb=False):
    def loss(preds, dtrain):
        # Normalize probs for LGB. Comment out for XGB
        if lgb:
            prob = 1 / (1 + np.exp(-preds))
        else:
            # Use the next line for XGB
            prob = preds
        prob = np.clip(prob, clip, 1 - clip)

        pos_mask = y_true == 1
        neg_mask = y_true == 0
        unl_mask = y_true == -1

        def gradient(p):
            return ((1 - p) ** (gamma)) * (gamma * p * np.log(p) + p - 1)

        def hess1(p):
            return((1 - p) ** (gamma)) * (
                    -gamma * (gamma - 1) * p ** 2 * np.log(p) - 2 * gamma * p * (p - 1) + (p - 1) ** 2)

        def hess2(p):
            return ((1 - p) ** (gamma)) * (1 - 2 * p) * (gamma * p * np.log(p) + p - 1)

        # Focal loss gradient and hessian per class
        grad_pos = gradient(prob)
        grad_neg = gradient(1 - prob)
        hess_pos = hess1(prob) + hess2(prob)
        hess_neg = hess1(1 - prob) + hess2(1 - prob)

        # --- PN Risk: supervised part ---
        grad_pn = np.zeros_like(preds)
        grad_pn[pos_mask] = pi_p * grad_pos[pos_mask]
        grad_pn[neg_mask] = (1 - pi_p) * (-grad_neg[neg_mask])
        hess_pn = np.zeros_like(preds)
        hess_pn[pos_mask] = pi_p * hess_pos[pos_mask]
        hess_pn[neg_mask] = (1 - pi_p) * hess_neg[neg_mask]
        # --- PU Risk: semi-supervised part ---
        grad_pu = np.zeros_like(preds)
        hess_pu = np.zeros_like(preds)
        grad_pu[pos_mask] = pi_p * grad_pos[pos_mask]
        hess_pu[pos_mask] = pi_p * hess_pos[pos_mask]

        # Non-negative PU correction
        pu_loss_unlabeled = -np.log(1 - prob[unl_mask]) * (prob[unl_mask] ** gamma)
        unl_risk = np.mean(pu_loss_unlabeled) if pu_loss_unlabeled.size > 0 else 0
        if unl_risk > 0:
            grad_pu[unl_mask] = -grad_neg[unl_mask]
            hess_pu[unl_mask] = hess_neg[unl_mask]

        # --- Final PNU Risk Combination ---
        grad_final = eta * grad_pn + (1 - eta) * grad_pu
        hess_final = eta * hess_pn + (1 - eta) * hess_pu
        hess_final = np.maximum(hess_final, 1e-6)  # Stability
        return grad_final, hess_final
    return loss


def nnpnu_focal_loss(y_true, gamma=3.0, eta=0.5, pi_p=0.3, clip=1e-7, lgb=False):
    """
    Focal PNU risk with the non-negative PU correction (nnPU), as a custom objective for LGB/XGB:
    R = eta * [pi * mean_P(l+) + (1 - pi) * mean_N(l-)]
        + (1 - eta) * [pi_U * mean_P(l+) + max(0, mean_U(l-) - pi_U * mean_P(l-))]
    P, N and U are the samples labeled 1, 0 and -1, l+ / l- the focal losses for the positive / negative class,
    and pi_U the positive rate in U: the hidden positives (pi * n - n_P) are assumed to be all in U.
    The risk is scaled by n, so the per-sample gradients and hessians are O(1) like in the other losses.
    :param y_true: PNU labels (1: positive, 0: reliable negative, -1: unlabeled)
    :param gamma: focusing parameter of the focal loss
    :param eta: weight of the PN risk against the PU risk
    :param pi_p: class prior P(y=1)
    """
    y_true = np.asarray(y_true)
    pos_mask = y_true == 1
    neg_mask = y_true == 0
    unl_mask = y_true == -1
    n = len(y_true)
    n_p, n_n, n_u = max(pos_mask.sum(), 1), max(neg_mask.sum(), 1), unl_mask.sum()
    pi_u = np.clip((pi_p * n - pos_mask.sum()) / n_u, 0, 1) if n_u > 0 else 0.0
    fl = FLLoss(r1=gamma)

    def loss(preds, dtrain):
        # Normalize probs for LGB. Comment out for XGB
        if lgb:
            prob = 1 / (1 + np.exp(-preds))
        else:
            # Use the next line for XGB
            prob = preds
        prob = np.clip(prob, clip, 1 - clip)

        # Focal loss of the negative class and the derivatives of l+ and l- w.r.t. the raw score
        l_neg = -(prob ** gamma) * np.log(1 - prob)
        grad_pos = fl.grad(prob, gamma)
        hess_pos = fl.hess1(prob, gamma) + fl.hess2(prob, gamma)
        grad_neg = -fl.grad(1 - prob, gamma)
        hess_neg = fl.hess1(1 - prob, gamma) + fl.hess2(1 - prob, gamma)

        # Non-negative PU correction (Kiryo et al., 2017): when the negative risk estimated from U
        # goes below zero, step against it (-grad) instead of fitting it further
        r_u = (np.mean(l_neg[unl_mask]) if n_u > 0 else 0.0) - pi_u * np.sum(l_neg[pos_mask]) / n_p
        sign = 1.0 if r_u >= 0 else -1.0

        # Per-sample weights of the risk (scaled by n)
        w_pos = n * (eta * pi_p + (1 - eta) * pi_u) / n_p
        w_neg = n * eta * (1 - pi_p) / n_n
        w_unl = n * (1 - eta) / max(n_u, 1)
        w_pos_as_neg = n * (1 - eta) * pi_u / n_p

        grad = np.zeros_like(prob)
        hess = np.zeros_like(prob)
        grad[pos_mask] = w_pos * grad_pos[pos_mask] - sign * w_pos_as_neg * grad_neg[pos_mask]
        grad[neg_mask] = w_neg * grad_neg[neg_mask]
        grad[unl_mask] = sign * w_unl * grad_neg[unl_mask]
        # Hessian: only the positive-curvature terms (the -pi_U * l- term on P is gradient-only)
        hess[pos_mask] = w_pos * hess_pos[pos_mask]
        hess[neg_mask] = w_neg * hess_neg[neg_mask]
        hess[unl_mask] = w_unl * hess_neg[unl_mask]
        hess = np.maximum(hess, 1e-6)  # Stability
        return grad, hess
    return loss
