import os
import sys
sys.path.append('..')
import numpy as np
import numpy.linalg as linalg
import scipy.optimize as op
from activation import *
from datamap import *
from prodata import *

class OriginNeuralNetwork:

    def __init__(self):
        pass

    def forward(self, params, hyperparams, X, y):
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

    def predict(self, params, hyperparams, X, y):
        """predict(params,x,y)
        x: the input of test data
        y: the output of test data
        params: it is a list of classifier params of each level.
        """
        res,_ = self.forward(params, hyperparams, X, y)
        return res


    def backward(self, params, hyperparams, X, y):
        """
        Implement the backward propagation presented in figure 2.

        Arguments:
        X -- input dataset, of shape (input size, number of examples)
        Y -- true "label" vector (containing 0 if cat, 1 if non-cat)
        cache -- cache output from forward_propagation()

        Returns:
        gradients -- A dictionary with the gradients with respect to each parameter, activation and pre-activation variables
        """

        Al,cache = self.forward(params,hyperparams,X,y)

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

    def gradient_descent(self, hyperparams, X, y):
        units = hyperparams['units']
        L = len(units)
        # the number of iterations
        iters = hyperparams['iters']
        alpha = hyperparams['alpha']
        params = init_params(units)
        while iters > 0 :
	        grads = self.backward(params,hyperparams,X,y)
            # update parameters with calculated gradients
	        for l in range(1,L):
	            params['W' + str(l)] -= alpha * grads['dW' + str(l)]
	            params['b' + str(l)] -= alpha * grads['db' + str(l)]
	        iters -= 1
        return params
