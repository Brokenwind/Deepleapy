import numpy as np
import re
from numpy.linalg import norm

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
    params = {}
    for i in np.arange(1,len(units)):
        lin = units[i-1]
        lout = units[i]
        W,b = debug_init_unit(lin,lout)
        params['W'+str(i)] = W
        params['b'+str(i)] = b

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
    params = {}
    # number of layers in the network
    L = len(units) 

    for l in range(1, L):
        params['W' + str(l)] = np.random.randn(units[l], units[l-1]) / np.sqrt(units[l-1])
        params['b' + str(l)] = np.zeros((units[l], 1))

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

def normdiff(dict1, dict2, prefix=None):
    """
    calculate the norm of the difference of two dataset
    """
    if prefix != None:
        dict1 = extract(dict1,prefix)
        dict2 = extract(dict2,prefix)

    flat1 = np.array([])
    flat2 = np.array([])

    for key in dict1.keys():
        flat1 = np.hstack((flat1,dict1[key].flatten()))
        flat2 = np.hstack((flat2,dict2[key].flatten()))

    diff = norm(flat1 - flat2)/norm(flat1 + flat2)

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
