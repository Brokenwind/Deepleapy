import numpy as np
import re
from numpy.linalg import norm

def init_empty(row, col):
    """
    Initialize a 2D array whose all the elements are np.ndarray
    """
    params = row*[np.array(col*[None])]
    params = np.array(params)
    # the first col need specical initialization
    for i in range(0, row):
        params[i,0] = np.array([])
    return params

def debug_init_unit(lin,lout):
    """
    Initialize the weights of a layer with lin incoming connections and 
    lout outgoing connections using a fixed strategy, 
    this will help you later in debugging
    """
    lin += 1
    """
    Initialize theta using "sin", this ensures that W is always of the same
    values and will be useful for debugging
    """
    theta = np.sin(np.arange(1,lin*lout+1))/10.0
    theta = theta.reshape((lout,lin))
    b = theta[:,0]
    b = b.reshape((b.size),1)
    W = np.delete(theta,0,axis=1)
    return W,b

def debug_init_params(units):
    """
    Initialize the parameters with fixed values for debugging

    Arguments:
    units -- python array (list) containing the dimensions of each layer in our network

    Returns:
    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
        Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
        bl -- bias vector of shape (layer_dims[l], 1)
    """
    params = 2*[np.array(len(units)*[None])]
    params = np.array(params)

    for i in np.arange(1,len(units)):
        lin = units[i-1]
        lout = units[i]
        W,b = debug_init_unit(lin,lout)
        params[0,i] = W
        params[1,i] = b

    return params

def init_params(units):
    """
    Arguments:
    units -- python array (list) containing the dimensions of each layer in our network

    Returns:
    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                    W1 -- weight matrix of shape (units[l], units[l-1])
                    b1 -- bias vector of shape (units[l], 1)
                    Wl -- weight matrix of shape (units[l-1], units[l])
                    bl -- bias vector of shape (1, units[l])

    Tips:
    - For example: the units for the "Planar Data classification model" would have been [2,2,1]. 
    This means W1's shape was (2,2), b1 was (1,2), W2 was (2,1) and b2 was (1,1). Now you have to generalize it!
    - In the for loop, use parameters['W' + str(l)] to access Wl, where l is the iterative integer.
    """

    np.random.seed(3)
    L = len(units) 
    params = init_empty(2,len(units))
    for l in range(1, L):
        params[0,l] = np.random.randn(units[l], units[l-1]) / np.sqrt(units[l-1])
        params[1,l] = np.zeros((units[l], 1))

    return params

def normalize(x,axis=0):
    """
    returns a normalized version of X where the mean value of each feature is 0 and the standard deviation is 1. This is often a good preprocessing step to do when working with learning algorithms.
    """
    mean = np.mean(x,axis)
    std = np.std(x,axis)

    if axis == 1:
        mean = mean.reshape((mean.size,1))
        std = std.reshape((std.size,1))
    else:
        mean = mean.reshape((1,mean.size))
        std = std.reshape((1,std.size))

    norm = (x - mean)/std

    return norm,mean,std

def extract(params,prefix):
    """
    extract values from a dict when the key has the same given prefix
    """
    pattern = re.compile(prefix)
    keys = params.keys()

    sels = []
    for key in keys:
        if pattern.match(key):
            sels.append(key)
    sels.sort()

    res = {}
    for key in sels:
        res[key] = params[key]

    return res

def normdiff(grad1, grad2):
    """
    calculate the norm of the difference of two dataset
    """
    if grad1.shape != grad2.shape:
        raise ValueError('The input grad1 and grad2 is not consistent')
    num = len(grad1)
    diff = np.array(num * [0.])
    for i in range(0,num):
        flat1 = np.array([])
        flat2 = np.array([])
        for l in range(1,len(grad1[i])):
            flat1 = np.hstack((flat1,grad1[i,l].flatten()))
            flat2 = np.hstack((flat2,grad2[i,l].flatten()))

        diff[i] = norm(flat1 - flat2)/norm(flat1 + flat2)

    return diff

def gen_batches(n, batch_size):
    """Generator to create slices containing batch_size elements, from 0 to n.
    The last slice may contain less than batch_size elements, when batch_size
    does not divide n.
    Examples
    --------
    >>> from sklearn.utils import gen_batches
    >>> list(gen_batches(7, 3))
    [slice(0, 3, None), slice(3, 6, None), slice(6, 7, None)]
    >>> list(gen_batches(6, 3))
    [slice(0, 3, None), slice(3, 6, None)]
    >>> list(gen_batches(2, 3))
    [slice(0, 2, None)]
    """
    start = 0
    for _ in range(int(n // batch_size)):
        end = start + batch_size
        yield slice(start, end)
        start = end
    if start < n:
        yield slice(start, n)

def train_test_split(X, y, **options):
    """Split arrays or matrices into train and test subsets
    Parameters
    ----------
    X : np.ndarray shape (n_features, n_samples)
    y : np.ndarray shape (n_classes, n_samples)/(n_samples,)/(1,n_samples)
    test_size : float, int, None, optional
        If float, should be between 0.0 and 1.0 and represent the proportion
        of the dataset to include in the test split.If None, the value is set
        to 0.25.
    shuffle : boolean, optional (default=True)
        Whether or not to shuffle the data before splitting. If shuffle=False
        then data is split in a stratified fashion
    Returns
    -------
    X_train :
    y_train :
    X_test  :
    y_test  :
    """
    if X is None or y is None:
        raise ValueError("Input data is None")
    X = np.array(X)
    y = np.array(y)
    if y.ndim == 1:
        y = y.reshape((1,y.size))
    if X.shape[1] != y.shape[1]:
        raise ValueError("X and y don't have the same number of samples")
    test_size = options.pop('test_size', None)
    shuffle = options.pop('shuffle', True)

    if options:
        raise TypeError("Invalid parameters passed: %s" % str(options))

    if test_size is None:
        test_size = 0.25

    # shuffle the data
    if shuffle:
        perm = np.random.permutation(y.shape[1])
        X = X[:,perm]
        y = y[:,perm]

    test_num = int(y.shape[1] * 0.25)

    return X[:,test_num:], y[:,test_num:], X[:,0:test_num], y[:,0:test_num]
