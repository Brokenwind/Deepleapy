import numpy as np

def debug_init_params(lin,lout):
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

def normalize(x):
    """
    returns a normalized version of X where the mean value of each feature is 0 and the standard deviation is 1. This is often a good preprocessing step to do when working with learning algorithms.
    """
    #if ( not isinstance(mean,np.ndarray) ) or ( not isinstance(mean,np.ndarray)):
    mean = np.mean(x,1)
    mean = mean.reshape((mean.size,1))
    std = np.std(x,1)
    std = std.reshape((std.size,1))
    norm = (x - mean)/std

    return norm,mean,std
