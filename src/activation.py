import numpy as np


def identity(X):
    """Simply return the input array.
    Parameters
    ----------
    X : {array-like, sparse matrix}, shape (n_features, n_samples)
        Data, where n_samples is the number of samples
        and n_features is the number of features.
    Returns
    -------
    X : {array-like, sparse matrix}, shape (n_features, n_samples)
        Same as the input data.
    """
    return X

def sigmoid(x):
    """
    Compute the sigmoid of x

    Parameters:
    X : A scalar or numpy array of any size.

    Returns
    s : sigmoid(x)
    """
    s = 1.0/(1.0+np.exp(-x))
    return s

def relu(x):
    """
    Compute the relu of x

    Parameters:
    X : A scalar or numpy array of any size.

    Returns
    s : relu(x)
    """
    s = np.maximum(0,x)

    return s

def tanh(X):
    """Compute the hyperbolic tan function inplace.
    Parameters
    ----------
    X : {array-like, sparse matrix}, shape (n_features, n_samples)
        The input data.
    Returns
    -------
    X_new : {array-like, sparse matrix}, shape (n_features, n_samples)
        The transformed data.
    """
    return np.tanh(X, out=X)


def softmax(x):
    """Compute the softmax function for each row of the input x.
    softmax(x)  = softmax(x+c)
    It is crucial that this function is optimized for speed because
    it will be used frequently in later code. You might find numpy
    functions np.exp, np.sum, np.reshape, np.max, and numpy
    broadcasting useful for this task.

    Parameters
    ----------
    x : {array-like, sparse matrix}, shape (n_features, n_samples)
        input data
    Returns:
    ----------
    x : {array-like, sparse matrix}
        softmax(x)
    """
    orig_shape = x.shape

    if len(x.shape) > 1:
        # Matrix

        # 1. Subtract the max value for solving overflow problem since we proved that softmax is invariat to constant offsets.
        tmp = np.max(x,axis=0,keepdims=True)
        x-=tmp

        # 2. compute the softmax
        x = np.exp(x)
        tmp = np.sum(x, axis = 0, keepdims=True)
        x /= tmp
    else:
        # Vector 
        tmp = np.max(x)
        x -= tmp
        x = np.exp(x)
        tmp = np.sum(x)
        x /= tmp
    return x

ACTIVATIONS = {'identity': identity, 'tanh': tanh, 'sigmoid': sigmoid,'relu': relu, 'softmax': softmax}

def identity_derivative(Z, delta):
    """Apply the derivative of the identity function: do nothing.
    Parameters
    ----------
    Z : {array-like, sparse matrix}, shape (n_features, n_samples)
        The data which was output from the identity activation function during
        the forward pass.
    delta : {array-like}, shape (n_features, n_samples)
         The backpropagated error signal to be modified inplace.
    """
    # Nothing to do

def sigmoid_derivative(Z, delta):
    """Apply the derivative of the logistic sigmoid function.
    It exploits the fact that the derivative is a simple function of the output
    value from logistic function.
    Parameters
    ----------
    Z : {array-like, sparse matrix}, shape (n_features, n_samples)
        The data which was output from the logistic activation function during
        the forward pass.
    delta : {array-like}, shape (n_features, n_samples)
         The backpropagated error signal to be modified inplace.
    """
    delta *= Z
    delta *= (1 - Z)


def tanh_derivative(Z, delta):
    """Apply the derivative of the hyperbolic tanh function.
    It exploits the fact that the derivative is a simple function of the output
    value from hyperbolic tangent.
    Parameters
    ----------
    Z : {array-like, sparse matrix}, shape (n_features, n_samples)
        The data which was output from the hyperbolic tangent activation
        function during the forward pass.
    delta : {array-like}, shape (n_features, n_samples)
         The backpropagated error signal to be modified inplace.
    """
    delta *= (1 - Z ** 2)


def relu_derivative(Z, delta):
    """Apply the derivative of the relu function.
    It exploits the fact that the derivative is a simple function of the output
    value from rectified linear units activation function.
    Parameters
    ----------
    Z : {array-like, sparse matrix}, shape (n_features, n_samples)
        The data which was output from the rectified linear units activation
        function during the forward pass.
    delta : {array-like}, shape (n_features, n_samples)
         The backpropagated error signal to be modified inplace.
    """
    delta[Z == 0] = 0


DERIVATIVES = {'identity': identity_derivative,
               'tanh': tanh_derivative,
               'sigmoid': sigmoid_derivative,
               'relu': relu_derivative}

if __name__ == '__main__':
    x = np.array([
        [9, 2, 5, 0, 0],
        [7, 5, 0, 0 ,0]])
    print("softmax(x) = " + str(softmax(x)))
    soft = ACTIVATIONS['softmax']
    print (soft(x))
