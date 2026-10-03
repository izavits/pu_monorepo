from __future__ import annotations

import numpy as np
from typing import Literal
from sklearn.metrics import ndcg_score, average_precision_score


def ndcg_at_k(
    y_true: np.ndarray,
    y_probas: np.ndarray,
    k: Literal["precision"] | int | float = "precision",
) -> float:
    """Generate ndcg at k depth. Simple wrapper around sklearn to facilitate the
       at precision value.

    Args:
        y_true (np.ndarray): {0,1} vector of labels.
        y_probas (np.ndarray): float vector of probabilities for the positive label.
        k (str, optional): Depth of recalled items.
                           - If int, this will be the depth to check.
                           - If float, it is interpreted as k% of the len(y_true)
                           - If "precision", it is recall-precision
                           Defaults to "precision".

    Returns:
        float: Recall @ K depth (or Recall-Precision)
    """
    # this is recall at precision level, aka number of positives
    num_to_check = 0
    if k == "precision":
        num_to_check = y_true.sum()
    else:
        # if not, check if it is int to report r@k
        if isinstance(k, int):
            num_to_check = k
        # else it is interpreted as r@k% of the test cases
        elif isinstance(k, float):
            num_to_check = int(k * len(y_true))
    num_to_check = min(num_to_check, len(y_true))
    return ndcg_score(y_true.reshape(1, -1), y_probas.reshape(1, -1), k=num_to_check)


def rec_at_k(
    y_true: np.ndarray,
    y_probas: np.ndarray,
    k: Literal["precision"] | int | float = "precision",
) -> float:
    """Generate recall at k depth.

    Args:
        y_true (np.ndarray): {0,1} vector of labels.
        y_probas (np.ndarray): float vector of probabilities for the positive label.
        k (str, optional): Depth of recalled items.
                           - If int, this will be the depth to check.
                           - If float, it is interpreted as k% of the len(y_true)
                           - If "precision", it is recall-precision
                           Defaults to "precision".

    Returns:
        float: Recall @ K depth (or Recall-Precision)
    """

    # this is recall at precision level, aka number of positives
    num_to_check = 0
    if k == "precision":
        num_to_check = y_true.sum()
    else:
        # if not, check if it is int to report r@k
        if isinstance(k, int):
            num_to_check = k
        # else it is interpreted as r@k% of the test cases
        elif isinstance(k, float):
            num_to_check = int(k * len(y_true))
    k = min(num_to_check, len(y_true))

    sorted_indices = np.argsort(y_probas)[::-1]
    found_labels = y_true[sorted_indices]
    return found_labels[:num_to_check].sum() / num_to_check


def get_ranking_scores(
    y_true: np.ndarray,
    y_probas: np.ndarray,
    k: Literal["precision"] | int | float = "precision",
) -> dict[str, float]:
    """Wrapper to get ranking scores together.

    Args:
        y_true (np.ndarray): {0,1} vector of labels.
        y_probas (np.ndarray): float vector of probabilities for the positive label.
        k (str, optional): Depth of recalled items.
                           - If int, this will be the depth to check.
                           - If float, it is interpreted as k% of the len(y_true)
                           - If "precision", it is recall-precision
                           Defaults to "precision".

    Returns:
        dict[str, float]:
    """
    res = {}
    res["average_precision"] = average_precision_score(y_true, y_probas)
    res["ndcg-precision"] = ndcg_at_k(y_true, y_probas)
    res["rec-precision"] = rec_at_k(y_true, y_probas)
    return res