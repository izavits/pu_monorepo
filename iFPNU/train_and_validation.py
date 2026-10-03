#
# train_and_validation.py
# Training loop and validation
#

from torch.optim import SGD
import copy
import torch
from numpy import vstack
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score, average_precision_score
from pu.evaluation import r_precision_at_n
from sklearn.metrics import precision_score, precision_recall_curve
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from tqdm import tqdm
import logging
import numpy as np
import matplotlib.pyplot as plt

logger = logging.getLogger('Main')


def train(train_dl, model, criterion, max_epochs=100, patience=5, log_interval=2,
          save_best_model=True, exp='test', run=None):
    """
    Training loop
    :param train_dl: the training data
    :param model: the model
    :param criterion: loss criterion
    :param max_epochs: maximum number of epochs
    :param patience: when to opt out from training if loss does not change
    :param log_interval: logging frequency
    :param save_best_model: whether to save best model
    :param exp: experiment name
    """
    # Define the optimization
    if criterion is None:
        raise NotImplementedError('Loss criterion is not defined.')
    optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
    running_loss = 0.0
    epoch_running_loss = 0.0
    waiting = 0
    min_loss = float('inf')
    best_model_dict = None
    for epoch in range(max_epochs):
        # Enumerate mini batches
        loop = tqdm(enumerate(train_dl), desc=f'Training epoch {epoch + 1}', colour='MAGENTA', ascii=' >=')
        for i, (inputs, targets) in loop:
            # Clear the gradients
            optimizer.zero_grad()
            # Compute the model output
            yhat = model(inputs)
            # Calculate loss
            loss = criterion(yhat, targets)
            running_loss += loss
            epoch_running_loss += loss
            if i % log_interval == log_interval - 1:
                loop.set_postfix_str(
                    f'[Epoch {epoch}, Minibatch {i + 1}] Loss: {(running_loss / log_interval):.5f}')
                running_loss = 0.0
            # Credit assignment
            loss.backward()
            # Update model weights
            optimizer.step()
        logger.info(f'==> Epoch {epoch} LOSS: {(epoch_running_loss / len(train_dl.dataset)):.5f}')
        if run is not None:
            run.log({"loss": epoch_running_loss / len(train_dl.dataset)})
        running_loss = 0.0
        if epoch_running_loss < min_loss:
            min_loss = running_loss
            best_model_dict = copy.deepcopy(model.state_dict())
            waiting = 0
        else:
            waiting += 1
            if waiting >= patience:
                break
        epoch_running_loss = 0.0
    if save_best_model:
        torch.save(best_model_dict, f"./out_models/{exp}")
    return best_model_dict


def validate(test_dl, model):
    """
    Validate model in unseen data
    :param test_dl: the test dataset
    :param model: the learned model
    """
    predictions, pred_proba, actuals = list(), list(), list()
    for i, (inputs, targets) in enumerate(test_dl):
        yhat = model(inputs)
        # Retrieve numpy array
        yhat = yhat.detach().numpy()
        actual = targets.numpy()
        actual = actual.reshape((len(actual), 1))
        pred_proba.append(yhat)
        # Round to class values
        yhat_round = yhat.round()
        # Store
        predictions.append(yhat_round)
        actuals.append(actual)
    predictions, pred_proba, actuals = vstack(predictions), vstack(pred_proba), vstack(actuals)
    # Calculate accuracy
    acc = accuracy_score(actuals, predictions)
    # Calculate AUC
    area = roc_auc_score(actuals, pred_proba)
    # Calculate R-precision
    rprecision = r_precision_at_n(pred_proba, actuals)
    # Calculate precision and recall
    precision = precision_score(actuals, predictions, average='binary')
    recall = recall_score(actuals, predictions, average='binary')
    # Calculate area under PR
    pr_area = average_precision_score(actuals, pred_proba)
    return acc, area, rprecision, precision, recall, pr_area


def f1_per_class(y_true, predictions):
    """
    F1 at the 0.5 threshold for the positive class, the negative class and their macro average
    :param y_true: the true labels
    :param predictions: the predicted labels (probabilities rounded at 0.5)
    :return: f1 for the positive class, f1 for the negative class, macro average f1
    """
    f1_neg, f1_pos = f1_score(y_true, predictions, labels=[0, 1], average=None, zero_division=0)
    f1_macro = (f1_pos + f1_neg) / 2
    return f1_pos, f1_neg, f1_macro


def f1_at_prior_threshold(y_true, scores, prior):
    """
    F1 for the positive class, the negative class and their macro average when the top `prior` fraction of
    the test scores is predicted positive. The threshold is the (1 - prior) quantile of the unlabeled test
    scores (transductive: no test labels are used to choose it)
    :param y_true: the true labels
    :param scores: the predicted scores (any monotone scale, e.g. probabilities)
    :param prior: the class prior P(y=1) used by the method
    :return: f1 for the positive class, f1 for the negative class, macro average f1, threshold
    """
    threshold = np.quantile(scores, 1 - prior)
    f1_pos, f1_neg, f1_macro = f1_per_class(y_true, (np.asarray(scores) >= threshold).astype(int))
    return f1_pos, f1_neg, f1_macro, threshold


def validate_sarem(x_test, y_test, model):
    pred_proba = model.predict_proba(x_test)[:,1]
    predictions = []
    for y_hat in pred_proba:
        yhat_round = y_hat.round()
        predictions.append(yhat_round)
    # Calculate accuracy
    acc = accuracy_score(y_test, predictions)
    # Calculate AUC
    area = roc_auc_score(y_test, pred_proba)
    # Calculate R-precision
    rprecision = r_precision_at_n(pred_proba, y_test)
    # Calculate precision and recall
    precision = precision_score(y_test, predictions, average='binary')
    recall = recall_score(y_test, predictions, average='binary')
    # Calculate area under PR
    pr_area = average_precision_score(y_test, pred_proba)

    # Plot PR curves
    p, r, thresholds = precision_recall_curve(y_test, pred_proba)
    # Find the threshold that gives the best F1-score
    numerator = 2 * r * p
    denom = r + p
    f1_scores = np.divide(numerator, denom, out=np.zeros_like(denom), where=(denom != 0))
    best_f1 = np.max(f1_scores)
    best_threshold_index = np.argmax(f1_scores)
    best_threshold = thresholds[best_threshold_index]
    print(f"Best Threshold: {best_threshold:.2f}")
    print(f"Best F1-score: {best_f1:.2f}")
    # Plot the precision-recall curve
    plt.figure(figsize=(8, 6))
    plt.plot(r, p, label='Precision-Recall curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='best')
    # Plot the threshold points
    plt.scatter(r[best_threshold_index], p[best_threshold_index], color='red',
                label=f'Best Threshold (F1 = {best_f1:.2f}, AP = {pr_area:.2f})')
    plt.legend(loc='best')
    # plt.show()
    # Calculate F1 at the 0.5 threshold for the positive class, the negative class and their average
    f1_pos, f1_neg, f1_macro = f1_per_class(y_test, predictions)
    return acc, area, rprecision, precision, recall, pr_area, best_f1, f1_pos, f1_neg, f1_macro, pred_proba, plt


def validate_tabpfn(x_test, y_test, model):
    pred_proba = model.predict_proba(x_test)[:,1]
    predictions = []
    for y_hat in pred_proba:
        yhat_round = y_hat.round()
        predictions.append(yhat_round)
    # Calculate accuracy
    acc = accuracy_score(y_test, predictions)
    # Calculate AUC
    area = roc_auc_score(y_test, pred_proba)
    # Calculate R-precision
    rprecision = r_precision_at_n(pred_proba, y_test)
    # Calculate precision and recall
    precision = precision_score(y_test, predictions, average='binary')
    recall = recall_score(y_test, predictions, average='binary')
    # Calculate area under PR
    pr_area = average_precision_score(y_test, pred_proba)

    # Plot PR curves
    p, r, thresholds = precision_recall_curve(y_test, pred_proba)
    # Find the threshold that gives the best F1-score
    numerator = 2 * r * p
    denom = r + p
    f1_scores = np.divide(numerator, denom, out=np.zeros_like(denom), where=(denom != 0))
    best_f1 = np.max(f1_scores)
    best_threshold_index = np.argmax(f1_scores)
    best_threshold = thresholds[best_threshold_index]
    print(f"Best Threshold: {best_threshold:.2f}")
    print(f"Best F1-score: {best_f1:.2f}")
    # Plot the precision-recall curve
    plt.figure(figsize=(8, 6))
    plt.plot(r, p, label='Precision-Recall curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='best')
    # Plot the threshold points
    plt.scatter(r[best_threshold_index], p[best_threshold_index], color='red',
                label=f'Best Threshold (F1 = {best_f1:.2f}, AP = {pr_area:.2f})')
    plt.legend(loc='best')
    # plt.show()
    return acc, area, rprecision, precision, recall, pr_area, best_f1, plt


def validate_xgb(x_test, y_test, model):
    """
    Validate the xgb model
    """
    pred_proba = model.predict(x_test)
    pred_proba = 1 / (1 + np.exp(-pred_proba))
    predictions = []
    for y_hat in pred_proba:
        yhat_round = y_hat.round()
        predictions.append(yhat_round)
    # Calculate accuracy
    acc = accuracy_score(y_test, predictions)
    # Calculate AUC
    area = roc_auc_score(y_test, pred_proba)
    # Calculate R-precision
    rprecision = r_precision_at_n(pred_proba, y_test)
    # Calculate precision and recall
    precision = precision_score(y_test, predictions, average='binary')
    recall = recall_score(y_test, predictions, average='binary')
    # Calculate area under PR
    pr_area = average_precision_score(y_test, pred_proba)

    # Plot PR curves
    p, r, thresholds = precision_recall_curve(y_test, pred_proba)
    # Find the threshold that gives the best F1-score
    numerator = 2 * r * p
    denom = r + p
    f1_scores = np.divide(numerator, denom, out=np.zeros_like(denom), where=(denom != 0))
    best_f1 = np.max(f1_scores)
    best_threshold_index = np.argmax(f1_scores)
    best_threshold = thresholds[best_threshold_index]
    print(f"Best Threshold: {best_threshold:.2f}")
    print(f"Best F1-score: {best_f1:.2f}")
    # Plot the precision-recall curve
    plt.figure(figsize=(8, 6))
    plt.plot(r, p, label='Precision-Recall curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='best')
    # Plot the threshold points
    plt.scatter(r[best_threshold_index], p[best_threshold_index], color='red',
                label=f'Best Threshold (F1 = {best_f1:.2f}, AP = {pr_area:.2f})')
    plt.legend(loc='best')
    #plt.show()
    # Calculate F1 at the 0.5 threshold for the positive class, the negative class and their average
    f1_pos, f1_neg, f1_macro = f1_per_class(y_test, predictions)
    return acc, area, rprecision, precision, recall, pr_area, best_f1, f1_pos, f1_neg, f1_macro, pred_proba, plt


def validate_gtt(model, cont_data, categ_data, target_data, device="cuda", val_batch_size=1):
    model = model.eval()
    results = np.zeros((categ_data.shape[0], 1))
    predictions = []

    for i in range(categ_data.shape[0] // val_batch_size):
        x_categ = torch.tensor(categ_data[val_batch_size * i:val_batch_size * i + val_batch_size]).to(
            dtype=torch.int64,
            device=device)
        x_cont = torch.tensor(cont_data[val_batch_size * i:val_batch_size * i + val_batch_size]).to(
            dtype=torch.float32,
            device=device)

        pred = model(x_categ, x_cont)
        results[val_batch_size * i:val_batch_size * i + val_batch_size, 0] = torch.sigmoid(
            pred).squeeze().cpu().detach().numpy()
    for i in results:
        y_hat = i.round()
        predictions.append(y_hat)
        # Calculate accuracy
    acc = accuracy_score(target_data[:results.shape[0]], predictions)
    # Calculate AUC
    area = roc_auc_score(target_data[:results.shape[0]], results[:, 0])
    # Calculate Rprecision
    rprecision = r_precision_at_n(results[:, 0], target_data[:results.shape[0]])
    # Calculate precision and recall
    precision = precision_score(target_data[:results.shape[0]], predictions, average='binary')
    recall = recall_score(target_data[:results.shape[0]], predictions, average='binary')
    # Calculate area under PR
    pr_area = average_precision_score(target_data[:results.shape[0]], results[:, 0])
    return acc, area, rprecision, precision, recall, pr_area


def train_gtt(
        model,
        criterion,
        optimizer,
        scheduler,
        train_cont,
        train_categ,
        train_target,
        val_cont,
        val_categ,
        val_target,
        device="cuda",
        batch_size=64,
        max_epochs=100,
        patience=10,
        save_best_model_dict=False,
        log_interval=10,
        exp='test'
):
    running_loss = 0.0
    max_score = 0
    best_model_dict = None
    waiting = 0
    for epoch in range(max_epochs):
        for i in range(train_categ.shape[0] // batch_size):
            optimizer.zero_grad()

            x_categ = torch.tensor(train_categ[batch_size * i:batch_size * i + batch_size]).to(
                dtype=torch.int64,
                device=device)
            x_cont = torch.tensor(train_cont[batch_size * i:batch_size * i + batch_size]).to(
                dtype=torch.float32,
                device=device)
            y_target = torch.tensor(train_target[batch_size * i:batch_size * i + batch_size]).to(
                dtype=torch.float32,
                device=device)
            pred = model(x_categ, x_cont)
            loss = criterion(pred, y_target)
            running_loss += loss
            loss.backward()
            optimizer.step()
            if i % log_interval == log_interval - 1:
                logger.info(f"epoch: {epoch + 1}, it: {i + 1}, loss: {running_loss / log_interval}")
                running_loss = 0.0
        running_loss = 0.0
        acc, current_score, rp, p, r, pr_area = validate_gtt(model, val_cont, val_categ, val_target,
                                                             device=device)
        # print("Validation score: ", current_score)
        scheduler.step()
        if current_score > max_score:
            max_score = current_score
            best_model_dict = copy.deepcopy(model.state_dict())
            waiting = 0
        else:
            waiting += 1
            if waiting >= patience:
                break
    if save_best_model_dict:
        torch.save(best_model_dict, f"./out_models/{exp}")
    return best_model_dict
