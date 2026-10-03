#
# evaluation.py
# Evaluation measures
#

import numpy as np


def rank_at_n(n, y_pred_proba, y_test):
    """
    Perform evaluation at top n results
    Calculate precision @n and recall rate @n
    :param n: Number of top results
    :param y_pred_proba: The array of predicted class probabilities for pos
    :param y_test: The array of the test classes
    :return: precision and recall
    """
    # Sort predicted probabilities for POS class and get the top indices
    sorted_y_pred_proba = np.argsort(y_pred_proba)
    top_n = sorted_y_pred_proba[::-1][:n]
    # Calculate precision @ n
    # Precision = correct fetched / n
    correct = 0
    for index in top_n:
        if y_test[index] == 1:
            correct += 1
    precision = float(correct / n)
    # Calculate recall rate @n
    # Recall = correct fetched / all in test set
    all_in_test_pos = 0
    for i in y_test:
        if i == 1:
            all_in_test_pos += 1
    recall = float(correct / all_in_test_pos)
    return precision, recall


def r_precision_at_n(y_pred_proba, y_test):
    """
    Calculate the r-precision (rR). This type of recall is actually defined
    as the precision at n, where n is the number of positive examples
    in the test set.
    :param y_pred_proba: The array of predicted probabilities
    :param y_test: The array of the test classes
    :return: r-precision at n
    """
    # Sanity check: the given n should equal the number of positives
    num_positives_in_test = 0
    for y in y_test:
        if y == 1:
            num_positives_in_test += 1
    precision, recall = rank_at_n(num_positives_in_test, y_pred_proba, y_test)
    return precision
