import numpy as np

def squared_loss(y_true, y_pred):
    """Compute the squared loss for regression.
    Parameters
    ----------
    y_true : array-like or label indicator matrix
        Ground truth (correct) values.
    y_pred : array-like or label indicator matrix
        Predicted values, as returned by a regression estimator.
    Returns
    -------
    loss : float
        The degree to which the samples are correctly predicted.
    """
    return ((y_true - y_pred) ** 2.).mean() / 2.


def log_loss(y_true, y_prob):
    """Compute Logistic loss for classification.
    It is loss function of softmax
    Parameters
    ----------
    y_true : array-like or label indicator matrix
        Ground truth (correct) labels.
    y_prob : array-like of float, shape = (n_samples, n_classes)
        Predicted probabilities, as returned by a classifier's
        predict_proba method.
    Returns
    -------
    loss : float
        The degree to which the samples are correctly predicted.
    """

    # to avoid devide by zero error
    y_prob = np.clip(y_prob, 1e-10, 1 - 1e-10)

    if y_prob.shape[1] == 1:
        y_prob = np.append(1 - y_prob, y_prob, axis=1)

    if y_true.shape[1] == 1:
        y_true = np.append(1 - y_true, y_true, axis=1)

    return -np.sum(y_true * np.log(y_prob)) / y_prob.shape[0]


def binary_log_loss(y_true, y_prob):
    """Compute binary logistic loss for classification.
    This is identical to log_loss in binary classification case,
    but is kept for its use in multilabel case.
    Parameters
    ----------
    y_true : array-like or label indicator matrix
        Ground truth (correct) labels.
    y_prob : array-like of float, shape = (n_samples, n_classes)
        Predicted probabilities, as returned by a classifier's
        predict_proba method.
    Returns
    -------
    loss : float
        The degree to which the samples are correctly predicted.
    """

    # to avoid devide by zero error
    y_prob = np.clip(y_prob, 1e-10, 1 - 1e-10)

    return -np.sum(y_true * np.log(y_prob) +
                   (1 - y_true) * np.log(1 - y_prob)) / y_prob.shape[0]

LOSS_FUNCTIONS = {'squared_loss': squared_loss, 'log_loss': log_loss,
                  'binary_log_loss': binary_log_loss}
