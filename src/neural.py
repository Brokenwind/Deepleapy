import os
import sys
sys.path.append('..')
import numpy as np
import numpy.linalg as linalg
import scipy.optimize as op
from activation import *


def compute_cost(params,hyperparams,x,y):
    """
    x: the input test data
    y: the label of relative x
    thetas: a list of all levels of estimated  value of unknown parameter
    reg: if it is True, means using regularized logistic. Default False
    lamda: it is used when reg=True
    """
    yh,cache = forward(params, hyperparams, x, y)
    # n: the number of class
    # m: the number row of result
    n,m = yh.shape
    # vectorize the real value and predicted result
    y = y.flatten()
    yh = yh.flatten()
    #the computation of cost function of each y(i)  is simmilar to the logistic regression cost function.
    #And you can compare it with the function compute_cost in ex2/optimlog.py
    all = y * np.log(yh) + (1 - y) * np.log(1-yh)
    J = -np.nansum(all)/m
    """
    if  isinstance(thetas,np.ndarray):
        thetas = reshapeList(thetas,units)
        for theta in thetas:
            # the first col of theta is not involed in calculation
            zero = np.zeros((np.size(theta,0),1))
            theta = np.hstack((zero,theta[:,1:]))
            theta = theta.flatten()
            J += lamda/(2.0*m)*(np.sum(theta * theta))
    """
    return J

def forward(params, hyperparams, X, y):
    """forward(thetas,x,y)
    x: the input of test data
    y: the output of test data
    thetas: it is a list of classifier params of each level.

    Return: the  final result a and all middle value z
    """
    units = hyperparams['units']
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
            A = relu(Z)
            #A = sigmoid(Z)
        else:
            A = sigmoid(Z)
        cache['Z'+str(l)] = Z
        cache['A'+str(l)] = A

    return A, cache

def predict(params, hyperparams, x, y):
    """predict(thetas,x,y)
    x: the input of test data
    y: the output of test data
    thetas: it is a list of classifier params of each level.
    """
    res,_ = forward(params, hyperparams, x, y)
    # col index of the max probality of each col
    pos = np.argmax(res,axis=0)
    # predicted values of each row of X
    pred = pos + 1

    return pred


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
            # np.int64(A > 0) is the gradient of ReLU
            dZ = dA * np.int64(Z > 0)
            """
            gz = sigmoid(Z)
            dZ = dA * gz * (1-gz)
            """
            dW = 1./m * np.dot(dZ, A.T)
            db = 1./m * np.sum(dZ, axis=1, keepdims = True)

        gradients['dZ'+str(l)] = dZ
        gradients['dW'+str(l)] = dW
        gradients['db'+str(l)] = db

    return gradients
