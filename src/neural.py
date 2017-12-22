import os
import sys
sys.path.append('..')
import numpy as np
import numpy.linalg as linalg
import scipy.optimize as op
from activation import *
from datamap import *

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


def numerical_gradient(params,hyperparams,x,y):
    check = 1e-4
    units = hyperparams['units']
    L = len(units)
    grads = {}

    # compute numerical gradient of parameter W
    for l in range(1,L):
        W = params['W'+str(l)]
        m,n = W.shape
        dW = np.zeros((m,n))
        for i in range(0,m):
            for j in range(0,n):
                tmp = W[i,j]
                W[i,j] = tmp + check
                up = compute_cost(params,hyperparams,x,y)
                W[i,j] = tmp - check
                down = compute_cost(params,hyperparams,x,y)
                dW[i,j] = (up - down)/(2.0*check)
                W[i,j] = tmp
        grads['dW'+str(l)] = dW

    # compute numerical gradient of parameter b
    for l in range(1,L):
        b = params['b'+str(l)]
        m,n = b.shape
        db = np.zeros((m,n))
        for i in range(0,m):
            for j in range(0,n):
                tmp = b[i,j]
                b[i,j] = tmp + check
                up = compute_cost(params,hyperparams,x,y)
                b[i,j] = tmp - check
                down = compute_cost(params,hyperparams,x,y)
                db[i,j] = (up - down)/(2.0*check)
                b[i,j] = tmp
        grads['db'+str(l)] = db

    return grads

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

def check_gradient(hyperparams=None, test_num = 5):

    deadline = 1e-8

    if hyperparams == None:
        hyperparams = {}

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

    params = {}
    for i in np.arange(1,L):
        lin = units[i-1]
        lout = units[i]
        W,b = debug_init_params(lin,lout)
        params['W'+str(i)] = W
        params['b'+str(i)] = b

    # Reusing debug_init_paramsializeWeights to generate X
    x,_ = debug_init_params( test_num, feature_num )
    # generate corresponding y
    y = np.mod(np.arange(1,test_num+1),class_num)
    y = map.class2matrix(y)
    # calculate the gradient with two diffent ways
    grad1 = backward(params,hyperparams,x,y)
    grad2 = numerical_gradient(params,hyperparams,x,y)

    # calculate the norm of the difference of two kinds of W
    fdW1 = np.array([])
    fdW2 = np.array([])
    for i in np.arange(1,L):
        W1 = grad1['dW'+str(i)]
        W2 = grad2['dW'+str(i)]
        fdW1 = np.hstack((fdW1,W1.flatten()))
        fdW2 = np.hstack((fdW2,W2.flatten()))
    diffW = linalg.norm(fdW1 - fdW2)/linalg.norm(fdW1 + fdW2)
    print ("Evaluate the norm of the difference between two W: %e" % (diffW))

    # calculate the norm of the difference of two kinds of b
    fdb1 = np.array([])
    fdb2 = np.array([])
    for i in np.arange(1,L):
        b1 = grad1['db'+str(i)]
        b2 = grad2['db'+str(i)]
        fdb1 = np.hstack((fdb1,b1.flatten()))
        fdb2 = np.hstack((fdb2,b2.flatten()))
    diffb = linalg.norm(fdb1 - fdb2)/linalg.norm(fdb1 + fdb2)
    print ("Evaluate the norm of the difference between two b: %e" % (diffb))

    res = ( diffW < deadline ) and ( diffb < deadline )
    if ( res ):
        print ("Passed the gradient check!")
    else:
        print ("Did not passed the gradient check!")
    return res
