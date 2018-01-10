import numpy as np

def check_consistent_length(*arrays):
    """Checks whether all objects in arrays have the same shape or length.
    Parameters
    ----------
    *arrays : list or tuple of 2D np.ndarray data.
        Objects that will be checked for consistent length.
    """
    shapes = [ X.shape for X in arrays if X is not None ]
    for i in range(1,len(shapes)):
        if shapes[i] != shapes[i-1]:
            raise ValueError("Found shapes of input variables are not inconsistent : %r" % shapes )

def accuracy_score(y_true, y_pred, normalize=True):
    """Accuracy classification score.
    the ratio of correct classification
    Parameters
    ----------
    y_true : array-like (features, samples)
        Ground truth (correct) labels.
    y_pred : array-like (features, samples)
        Predicted labels, as returned by a classifier.
    normalize : bool, optional (default=True)
        If ``False``, return the number of correctly classified samples.
        Otherwise, return the fraction of correctly classified samples.
    Returns
    -------
    score : float
        If ``normalize == True``, return the correctly classified samples
        (float), else it returns the number of correctly classified samples
        (int).
        The best performance is 1 with ``normalize == True`` and the number
        of samples with ``normalize == False``.
    """

    # Compute accuracy for each possible representation
    check_consistent_length(y_true, y_pred)
    if y_true.ndim == 1 or y_true.shape[0] == 1 or y_true.shape[1] == 1:
        num = y_true.size
        y_pred[y_pred >= 0.5] = 1
        y_pred[y_pred < 0.5] = 0
        score = 1.0*np.sum(y_true == y_pred)
    else:
        num = y_true.shape[1]
        idx1 = np.argmax(y_true,axis=0)
        idx2 = np.argmax(y_pred,axis=0)
        score = 1.0*np.sum(idx1 == idx2)
    if normalize:
        score = 1.0*score / num

    return score

def r2_score(y_true, y_pred, axis=0):
    """R^2 (coefficient of determination) regression score function.
    Best possible score is 1.0 and it can be negative (because the
    model can be arbitrarily worse). A constant model that always
    predicts the expected value of y, disregarding the input features,
    would get a R^2 score of 0.0.
    Read more in the :ref:`User Guide <r2_score>`.
    Parameters
    ----------
    y_true : ndarray of shape = (n_samples) or (n_outputs,n_samples)
        Ground truth (correct) target values.
    y_pred : ndarray of shape = (n_samples) or (n_outputs,n_samples)
        Estimated target values.
    Returns
    -------
    z : float or ndarray of floats
        The R^2 score or ndarray of scores if 'multioutput' is
        'raw_values'.
    Notes
    -----
    This is not a symmetric function.
    Unlike most other scores, R^2 score may be negative (it need not actually
    be the square of a quantity R).
    References
    ----------
    .. [1] `Wikipedia entry on the Coefficient of determination
            <https://baike.baidu.com/item/%E5%8F%AF%E5%86%B3%E7%B3%BB%E6%95%B0/8020809?fr=aladdin&fromid=18081717&fromtitle=coefficient+of+determination>`_
    """
    if y_true.ndim == 1:
        if axis == 1:
            y_true = y_true.reshape((1,y_true.size))
            y_pred = y_pred.reshape((1,y_pred.size))
            # simplify the calculation
            y_true = y_true.T
            y_pred = y_pred.T
        else:
            y_true = y_true.reshape((y_true.size, 1))
            y_pred = y_pred.reshape((y_pred.size, 1))

    check_consistent_length(y_true, y_pred)
    numerator = ( (y_true - y_pred) ** 2).sum(axis=0, dtype=np.float64)
    denominator = ((y_true - np.average(y_true, axis=0)) ** 2).sum(axis=0, dtype=np.float64 )
    nonzero_denominator = denominator != 0
    nonzero_numerator = numerator != 0
    valid_score = nonzero_denominator & nonzero_numerator
    output_scores = np.ones([y_true.shape[1]])
    output_scores[valid_score] = 1 - (numerator[valid_score] /
                                      denominator[valid_score])
    # arbitrary set to zero to avoid -inf scores, having a constant
    # y_true is not interesting for scoring a regression anyway
    output_scores[nonzero_numerator & ~nonzero_denominator] = 0.

    return np.average(output_scores)

if __name__ == '__main__':
    print(accuracy_score(np.array([[0, 1], [1, 1]]), np.ones((2, 2))))
    #print(accuracy_score(np.array([0, 1]), np.ones((1, 2))))
    y_true = np.array([3, -0.5, 2, 7])
    y_pred = np.array([2.5, 0.0, 2, 8])
    print (r2_score(y_true, y_pred))
    y_true = np.array([[0.5, 1], [-1, 1], [7, -6]])
    y_pred = np.array([[0, 2], [-1, 2], [8, -5]])
    print (r2_score(y_true.T, y_pred.T))
    y_true = np.array([1,2,3])
    y_pred = np.array([1,2,3])
    print (r2_score(y_true, y_pred))
    y_true = np.array([1,2,3])
    y_pred = np.array([3,2,1])
    print (r2_score(y_true, y_pred))
