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


def softmax_loss(y_true, y_prob, axis=0, clip=False):
    """Compute Logistic loss for classification.
    It is loss function of softmax
    Parameters
    ----------
    y_true : array-like or label indicator matrix
        Ground truth (correct) labels.
    y_prob : array-like of float, shape = (n_samples, n_classes)
        Predicted probabilities, as returned by a classifier's
        predict_proba method.
    axis : which axis means the nubmer of samples
    clip : will you avoid  error devided by zeor. it will set y_prob = 1e-10 when y_prob < 1e-10, and y_prob = 1-1e-10 when y_prob > 1-1e-10

    Returns
    -------
    loss : float
        The degree to which the samples are correctly predicted.
    """

    # to avoid devided by zero error
    if clip:
        y_prob = np.clip(y_prob, 1e-10, 1 - 1e-10)

    if y_prob.shape[1] == 1:
        y_prob = np.append(1 - y_prob, y_prob, axis=1)

    if y_true.shape[1] == 1:
        y_true = np.append(1 - y_true, y_true, axis=1)

    return -np.sum(y_true * np.log(y_prob)) / y_prob.shape[0]


def log_loss(y_true, y_prob, axis=0,clip=False):
    """Compute binary logistic loss for classification.
    This is identical to softmax_loss in binary classification case,
    but is kept for its use in multilabel case.
    Parameters
    ----------
    y_true : array-like or label indicator matrix
        Ground truth (correct) labels.
    y_prob : array-like of float, shape = (n_samples, n_classes)
        Predicted probabilities, as returned by a classifier's
        predict_proba method.
    axis : which axis means the nubmer of samples
    clip : will you avoid  error devided by zeor. it will set y_prob = 1e-10 when y_prob < 1e-10, and y_prob = 1-1e-10 when y_prob > 1-1e-10
    Returns
    -------
    loss : float
        The degree to which the samples are correctly predicted.
    """

    # to avoid devided by zero error
    if clip:
        y_prob = np.clip(y_prob, 1e-10, 1 - 1e-10)

    n_samples = 0
    if y_prob.ndim == 1:
        n_samples = y_prob.size
    else:
        if axis == 0 :
            n_samples = y_prob.shape[0]
        else:
            n_samples = y_prob.shape[1]

    res = -np.sum(y_true * np.log(y_prob) +
                  (1 - y_true) * np.log(1 - y_prob)) / n_samples

    return res


LOSS_FUNCTIONS = {'squared_loss': squared_loss, 'softmax_loss': softmax_loss,
                  'log_loss': log_loss}

def compute_cost(params, hyperparams, y_true, y_prob):
    """
    X: the input test data
    y: the label of relative x
    params: a list of all levels of estimated  value of unknown parameter
    reg: if it is True, means using regularized logistic. Default False
    lamda: it is used when reg=True
    """
    # the regularition parameter
    units = hyperparams['units']
    lamd = hyperparams['lamda']
    loss = hyperparams['lossfunc']

    L = len(units)

    # n: the number of class
    # m: the number row of result
    n,m = y_prob.shape

    lossfunc = LOSS_FUNCTIONS[loss]

    J = lossfunc(y_true,y_prob,axis=1)

    regular = 0
    for l in range(1,L):
        W = params['W' + str(l)]
        b = params['b' + str(l)]
        regular += np.sum(W * W)

    J += lamd/(2.0*m)*regular

    return J
