#
# pu_data.py
# Utils to make pu data
#

import math
import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_consistent_length


def create_pu_data(X, y, ratio=None, label_assumption='SCAR', n_neighbors=None, random_state=123):
    """
    Create PU data by creating a hidden percentage of positives into negatives
    :param X: samples (n_samples, n_features)
    :param y: labels (n_samples, 1)
    :param ratio: the ratio of hidden positives
    :param label_assumption: the labelling mechanism (SCAR, SAR)
    :param n_neighbors: the number of KNN for SAR assumption
    :return: the "noisy" labels y_
    """
    # Make a copy of the correct labels
    y_ = y.copy()
    check_consistent_length(X, y_)
    # Calculate number of positives
    n_pos = np.sum(y_ == 1)
    random_state = check_random_state(random_state)

    if ratio is None:
        # No PU data, just the original PN labels
        return y_
    if n_neighbors is None:
        n_neighbors = math.floor(math.sqrt(len(y_)))
    if label_assumption == 'SCAR':
        # Uniformly selected positives
        p_ = None
    elif label_assumption == 'SAR':
        # Positives selected at random based on position in space
        dist_matrix = pairwise_distances(X, n_jobs=-1)
        # Positive and negative indexes
        ix_pos = np.where(y == 1)[0]
        ix_neg = np.where(y == 0)[0]
        pos_dist_mt = dist_matrix[ix_pos][:, ix_neg]
        k_mean_dist = np.mean(pos_dist_mt[:, :n_neighbors], axis=1)
        sample_weight = np.array([k_mean_dist[i] / np.sum(k_mean_dist) for i, _ in enumerate(k_mean_dist)])
        p_ = sample_weight.reshape(-1, )
    else:
        raise ValueError('Labeling mechanism not valid; Use SCAR or SAR.')

    # Size of hidden positives
    if label_assumption == 'SAR':
        size_ = min(int(np.ceil(ratio * n_pos)), len(np.nonzero(p_)[0]))
    else:
        size_ = int(np.ceil(ratio * n_pos))
    indices = random_state.choice(range(n_pos), size_, replace=False, p=p_)
    pos_y = y_[y_ == 1].copy()
    pos_y[indices] = 0
    y_[y_ == 1] = pos_y
    return y_
