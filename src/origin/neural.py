import os
import sys
sys.path.append('..')
import numpy as np
import numpy.linalg as linalg
import scipy.optimize as op
from activation import *
from datamap import *
from prodata import *
from loss import LOSS_FUNCTIONS

def compute_cost(params,hyperparams,X,y):
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

    yh,cache = forward(params, hyperparams, X, y)
    # n: the number of class
    # m: the number row of result
    n,m = yh.shape

    lossfunc = LOSS_FUNCTIONS[loss]

    J = lossfunc(y,yh,axis=1)

    regular = 0
    for l in range(1,L):
        W = params['W' + str(l)]
        b = params['b' + str(l)]
        regular += np.sum(W * W)

    J += lamd/(2.0*m)*regular

    return J

def forward(params, hyperparams, X, y):
    """forward(params, hyperparams, X, y)
    X: the input of test data
    y: the output of test data
    params: it is a list of classifier params of each level.

    Return: the  final result a and all middle value z
    """
    units = hyperparams['units']
    # activition function
    activition = hyperparams['activition']
    L = len(units)
    cache={}
    # LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID
    A = X.copy()
    cache['A0'] = A
    for l in range(1, L):
        W = params['W' + str(l)]
        b = params['b' + str(l)]
        Z = np.dot(W, A) + b
        if l != L - 1:
            if activition == 'relu':
                A = relu(Z)
            else:
                A = sigmoid(Z)
        # the last layer is using sigmoid to do classification
        else:
            A = sigmoid(Z)
        cache['Z'+str(l)] = Z
        cache['A'+str(l)] = A

    return A, cache

def predict(params, hyperparams, X, y):
    """predict(params,x,y)
    x: the input of test data
    y: the output of test data
    params: it is a list of classifier params of each level.
    """
    res,_ = forward(params, hyperparams, X, y)
    return res


def backward(params,hyperparams,X, y):
    """
    Implement the backward propagation presented in figure 2.

    Arguments:
    X -- input dataset, of shape (input size, number of examples)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat)
    cache -- cache output from forward_propagation()

    Returns:
    gradients -- A dictionary with the gradients with respect to each parameter, activation and pre-activation variables
    """

    Al,cache = forward(params,hyperparams,X,y)

    m = X.shape[1]

    units = hyperparams['units']
    # activition function
    activition = hyperparams['activition']
    lamd = hyperparams['lamda']
    L = len(units)
    gradients = {}

    for l in range(L-1,0,-1):
        Z = cache['Z'+str(l)]
        A = cache['A'+str(l-1)]
        if l == L-1 :
            dZ = Al - y
            dW = 1./m * np.dot(dZ, A.T)
            db = 1./m * np.sum(dZ, axis=1, keepdims = True)
        else:
            # use W in previous layer
            Wp = params['W'+str(l+1)]
            # use calculated dZ in previous layer
            dA = np.dot(Wp.T, dZ)
            if activition == 'relu':
                # np.int64(A > 0) is the gradient of ReLU
                dZ = dA * np.int64(Z > 0)
            else:
                gZ = sigmoid(Z)
                dZ = dA * gZ * (1-gZ)

            dW = 1./m * np.dot(dZ, A.T)
            db = 1./m * np.sum(dZ, axis=1, keepdims = True)
        # add regularition item
        dW += 1.0*lamd/m * params['W'+str(l)]

        gradients['dZ'+str(l)] = dZ
        gradients['dW'+str(l)] = dW
        gradients['db'+str(l)] = db

    return gradients


def gradient_descent(hyperparams,X,y):
    units = hyperparams['units']
    L = len(units)
    # the number of iterations
    iters = hyperparams['iters']
    alpha = hyperparams['alpha']

    params = init_params(units)
    while iters > 0 :
        grads = backward(params,hyperparams,X,y)
        # update parameters with calculated gradients
        for l in range(1,L):
            params['W' + str(l)] -= alpha * grads['dW' + str(l)]
            params['b' + str(l)] -= alpha * grads['db' + str(l)]
        iters -= 1

    return params

def numerical_gradient_part(params,hyperparams,X,y,prefix):
    """
    compute numerical gradient of parameter specified with prefix
    """
    check = 1e-4
    grads={}
    units = hyperparams['units']
    for l in range(1,len(units)):
        value = params[prefix+str(l)]
        m,n = value.shape
        dTmp = np.zeros((m,n))
        for i in range(0,m):
            for j in range(0,n):
                tmp = value[i,j]
                value[i,j] = tmp + check
                up = compute_cost(params,hyperparams,X,y)
                value[i,j] = tmp - check
                down = compute_cost(params,hyperparams,X,y)
                dTmp[i,j] = (up - down)/(2.0*check)
                value[i,j] = tmp

        grads['d'+prefix+str(l)] = dTmp

    return grads

def numerical_gradient(params,hyperparams,X,y):
    """
    Compute all the numerical gradients of parameters in cost function
    Arguments:
    params -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3":
    hyperparams -- python dictionary containing your hyperparameters

    X -- input datapoint, of shape (input size, 1)
    y -- true "label"

    Returns:
    grads -- numerical gradients of the cost function

    """
    # compute numerical gradient of parameter W
    gradW = numerical_gradient_part(params,hyperparams,X,y,'W')
    # compute numerical gradient of parameter b
    gradb = numerical_gradient_part(params,hyperparams,X,y,'b')
    # merge the two parts of the gradient
    grads = dict(gradW,**gradb)

    return grads


def check_gradient(hyperparams=None, test_num = 5):
    """
    Checks if backward propagation computes correctly the gradient of the cost function

    Arguments:
    hyperparams -- python dictionary containing your hyperparameters
    test_num -- how many test samples you will use
    Returns:
    difference -- difference between the approximated gradient and the backward propagation gradient

    """

    deadline = 1e-8

    if hyperparams == None:
        hyperparams = {}

    hyperparams['activition'] = 'sigmoid'
    hyperparams['lamda'] = 1.
    hyperparams['lossfunc'] = 'log_loss'

    if 'units' in hyperparams.keys():
        units = hyperparams['units']
    else:
        # default number units of each layer
        units = [4,5,3]
        hyperparams['units'] = units

    L = len(units)
    # the number of ouput classifications
    class_num = units[L-1]
    # the number of features of input data
    feature_num = units[0]
    map = DataMap(range(0,class_num))

    params = debug_init_params(units)

    # Reusing debug_init_paramsializeWeights to generate X
    X,_ = debug_init_unit( test_num, feature_num )
    # generate corresponding y
    y = np.mod(np.arange(1,test_num+1),class_num)
    y = map.class2matrix(y)
    # calculate the gradient with two diffent ways
    grad1 = backward(params,hyperparams,X,y)
    grad2 = numerical_gradient(params,hyperparams,X,y)

    # calculate the norm of the difference of two kinds of W
    diffW = normdiff(grad1,grad2,prefix='dW')
    print ("Evaluate the norm of the difference between two dW: %e" % (diffW))

    # calculate the norm of the difference of two kinds of b
    diffb = normdiff(grad1,grad2,prefix='db')
    print ("Evaluate the norm of the difference between two db: %e" % (diffb))

    res = ( diffW < deadline ) and ( diffb < deadline )
    if ( res ):
        print ("Passed the gradient check!")
    else:
        print ("Did not passed the gradient check!")

    return res
