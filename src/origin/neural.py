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
    """
    units : tuple, length = n_layers - 2, default (100,)
        The ith element represents the number of neurons in the ith
        hidden layer.
    activation : {'identity', 'sigmoid', 'tanh', 'relu'}, default 'relu'
        Activation function for the hidden layer.
        - 'identity', no-op activation, useful to implement linear bottleneck,
          returns f(x) = x
        - 'sigmoid', the sigmoid function,
          returns f(x) = 1 / (1 + exp(-x)).
        - 'tanh', the hyperbolic tan function,
          returns f(x) = tanh(x).
        - 'relu', the rectified linear unit function,
          returns f(x) = max(0, x)
    solver : {'bgd', 'sgd', 'mbgd'}, default 'adam'
        The solver for weight optimization.
        - 'bgd' batch gradient descent
        - 'sgd' refers to stochastic gradient descent.
        - 'minibatch' mini batch gradient descent
    L2_penalty : float, optional, default 0.0001
        L2 penalty (regularization term) parameter.
    batch_size : int, optional, default 'auto'
        Size of minibatches for stochastic optimizers.
        If the solver is 'lbfgs', the classifier will not use minibatch.
        When set to "auto", `batch_size=min(200, n_samples)`
    learning_rate_type : {'constant', 'invscaling', 'adaptive'}, default 'constant'
        Learning rate schedule for weight updates.
        - 'constant' is a constant learning rate given by
          'learning_rate_init'.
        - 'invscaling' gradually decreases the learning rate ``learning_rate_``
          at each time step 't' using an inverse scaling exponent of 'power_t'.
          effective_learning_rate = learning_rate_init / pow(t, power_t)
        - 'adaptive' keeps the learning rate constant to
          'learning_rate_init' as long as training loss keeps decreasing.
          Each time two consecutive epochs fail to decrease training loss by at
          least tol, or fail to increase validation score by at least tol if
          'early_stopping' is on, the current learning rate is divided by 5.
        Only used when solver='sgd'.
    learning_rate_init : double, optional, default 0.001
        The initial learning rate used. It controls the step-size
        in updating the weights. Only used when solver='sgd' or 'adam'.
    max_iters : int, optional, default 200
        Maximum number of iterations. The solver iterates until convergence
        (determined by 'tol') or this number of iterations. For stochastic
        solvers ('sgd', 'adam'), note that this determines the number of epochs
        (how many times each data point will be used), not the number of
        gradient steps.
    tol : float, optional, default 1e-4
        Tolerance for the optimization. When the loss or score is not improving
        by at least ``tol`` for ``n_iter_no_change`` consecutive iterations,
        unless ``learning_rate`` is set to 'adaptive', convergence is
        considered to be reached and training stops.
    """

    hyperparams = {}

    def __init__(self,units,model_type='classification',solver='bgd',lossfunc='log_loss',activation="relu",out_activation='sigmoid',learning_rate_type='constant',learning_rate_init=0.01,L2_penalty=0.01,max_iters=200,tol=1e-4):
        self.hyperparams['units'] = units
        self.hyperparams['model_type'] = model_type
        self.hyperparams['lossfunc'] = lossfunc
        self.hyperparams['activation'] = activation
        self.hyperparams['out_activation'] = out_activation
        self.hyperparams['learning_rate_type'] = learning_rate_type
        self.hyperparams['learning_rate_init'] = learning_rate_init
        self.hyperparams['L2_penalty'] = L2_penalty
        self.hyperparams['max_iters'] = max_iters
        self.hyperparams['tol'] = tol

    def get_hyperparams(self):
        """
        Return the current hyperparameters
        """
        return self.hyperparams

    def set_hyperparams(self, hyperparams):
        """
        Change the part of current hyperparameters with given hyperparameters
        """
        for key in hyperparams.keys():
            self.hyperparams[key] = hyperparams[key]

    def forward(self, params, X, y):
        """forward(params, self.hyperparams, X, y)
        X: the input of test data
        y: the output of test data
        params: it is a list of out_activation params of each level.

        Return: the  final result a and all middle value z
        """
        units = self.hyperparams['units']
        # activation function
        activation = self.hyperparams['activation']
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
                if activation == 'relu':
                    A = relu(Z)
                else:
                    A = sigmoid(Z)
		            # the last layer is using sigmoid to do classification
            else:
                A = sigmoid(Z)

            cache['Z'+str(l)] = Z
            cache['A'+str(l)] = A

        return A, cache

    def predict(self, params, X, y):
        """predict(params,x,y)
        x: the input of test data
        y: the output of test data
        params: it is a list of out_activation params of each level.
        """
        res,_ = self.forward(params, X, y)
        return res


    def backward(self, params, X, y):
        """
        Implement the backward propagation presented in figure 2.

        Arguments:
        X -- input dataset, of shape (input size, number of examples)
        Y -- true "label" vector (containing 0 if cat, 1 if non-cat)
        cache -- cache output from forward_propagation()

        Returns:
        gradients -- A dictionary with the gradients with respect to each parameter, activation and pre-activation variables
        """

        Al,cache = self.forward(params,X,y)

        m = X.shape[1]

        units = self.hyperparams['units']
        # activation function
        activation = self.hyperparams['activation']
        L2_penalty = self.hyperparams['L2_penalty']
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
                if activation == 'relu':
                    # np.int64(A > 0) is the gradient of ReLU
                    dZ = dA * np.int64(Z > 0)
                else:
                    gZ = sigmoid(Z)
                    dZ = dA * gZ * (1-gZ)

                dW = 1./m * np.dot(dZ, A.T)
                db = 1./m * np.sum(dZ, axis=1, keepdims = True)

            # add regularition item
            dW += 1.0*L2_penalty/m * params['W'+str(l)]

            gradients['dZ'+str(l)] = dZ
            gradients['dW'+str(l)] = dW
            gradients['db'+str(l)] = db

        return gradients

    def gradient_descent(self, X, y):
        units = self.hyperparams['units']
        L = len(units)
        # the number of iterations
        max_iters = self.hyperparams['max_iters']
        learning_rate_init = self.hyperparams['learning_rate_init']
        params = init_params(units)
        while max_iters > 0 :
	        grads = self.backward(params,X,y)
            # update parameters with calculated gradients
	        for l in range(1,L):
	            params['W' + str(l)] -= learning_rate_init * grads['dW' + str(l)]
	            params['b' + str(l)] -= learning_rate_init * grads['db' + str(l)]
	        max_iters -= 1
        return params
