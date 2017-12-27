import os
import sys
sys.path.append('..')
import numpy as np
import numpy.linalg as linalg
import scipy.optimize as op
from activation import *
from datamap import *
from prodata import *
from loss import compute_cost
from base import Classifier

class OriginNeuralNetwork(Classifier):
    """
    classes : tuple, list, np.ndarray
        This constains the different classes of you labels. len(classes) should
        be equal to the last value of units(except when it is 1 meaning binary classification )
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
    solver : {'BGD','MBGD'}, default 'adam'
        The solver for weight optimization.
        - 'BGD' batch gradient descent
        - 'MBGD' mini batch gradient descent
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
        Only used when solver='MBGD'.
    learning_rate_init : double, optional, default 0.001
        The initial learning rate used. It controls the step-size
        in updating the weights. Only used when solver='MBGD' or 'adam'.

    max_iters : int, optional, default 200
        Maximum number of iterations. The solver iterates until convergence
        (determined by 'tol') or this number of iterations. For stochastic
        solvers ('MBGD'), note that this determines the number of epochs
        (how many times each data point will be used), not the number of
        gradient steps.
    tol : float, optional, default 1e-4
        Tolerance for the optimization. When the loss or score is not improving
        by at least ``tol`` for ``n_iter_no_change`` consecutive iterations,
        unless ``learning_rate`` is set to 'adaptive', convergence is
        considered to be reached and training stops.
    batch_size :
    no_improve_num : int, optional, default 10
        Maximum number of epochs to not meet ``tol`` improvement.
        Only effective when solver='sgd' or 'adam'
    early_stopping : 
    """

    hyperparams = {}
    params = {}
    # how many layers of the neural network
    layers = 0
    # the number of iterations
    iters_count = 0
    # the number of epochs to not meet ``tol`` improvement.
    no_improve_count = 0
    # cost values of on every updated parameters
    costs = []
    # the best(minimal) cost value of all the current cost values
    best_cost = 1e10
    validation_scores = []
    best_validation_score = 0.0
    # it is the instance of DataMap, which will tranform between names of classes and its indexes

    def __init__(self,units,
                 model_type='classification',solver='BGD',
                 lossfunc='log_loss',activation="relu",out_activation='sigmoid',
                 learning_rate_type='constant',learning_rate_init=0.01,
                 L2_penalty=0.01,max_iters=200, batch_size=64, tol=1e-4,
                 verbose=True, no_improve_num=10, early_stopping=False):

        self.layers = len(units)

        self.hyperparams['units'] = units
        self.hyperparams['solver'] = solver
        self.hyperparams['model_type'] = model_type
        self.hyperparams['lossfunc'] = lossfunc
        self.hyperparams['activation'] = activation
        self.hyperparams['out_activation'] = out_activation
        self.hyperparams['learning_rate_type'] = learning_rate_type
        self.hyperparams['learning_rate_init'] = learning_rate_init
        self.hyperparams['L2_penalty'] = L2_penalty
        self.hyperparams['max_iters'] = max_iters
        self.hyperparams['batch_size'] = batch_size
        self.hyperparams['tol'] = tol
        self.hyperparams['no_improve_num'] = no_improve_num
        self.hyperparams['early_stopping'] = early_stopping


    def get_hyperparams(self):
        """
        Return the current hyperparamsarameters
        """
        return self.hyperparams.copy()

    def get_costs(self):
        """
        Return the current all the cost values
        """
        # list.copy() only exists in Python3
        return costs.copy()

    def get_params(self):
        """
        Get the current parameters
        """
        return self.params.copy()

    def set_params(self,params):
        """
        Set you own parameters to neural network
        """
        self.params = params.copy()

    def set_hyperparams(self, hyperparams):
        """
        Change the part of current hyperparamsarameters with given hyperparamsarameters
        """
        for key in hyperparams.keys():
            if key in self.hyperparams.keys():
                self.hyperparams[key] = hyperparams[key]
            else:
                raise ValueError('The "%s" is not a hyperparam' % (key))
        self.layers = len(self.hyperparams['units'])

    def forward(self, X, params=None):
        """Forward propatation of neural network
        Parameters
        ----------
        X : np.ndarray (features, samples)
            the input of test data
        y : np.ndarray (classes, samples)
            the output of test data
        params : python dictionary (if it is not set, the function will use self.params instead)
            python dictionary containing your parameters "Wl", "bl"(l means the lth layer)
        Returns
        -------
        A : the result of last layer
        cache : "Wl", "bl"(l means the lth layer)of all the layers
        """
        if params is not None:
            self.params = params
        # activation function
        activation = self.hyperparams['activation']
        cache={}
        A = X.copy()
        cache['A0'] = A
        for l in range(1, self.layers):
            W = self.params['W' + str(l)]
            b = self.params['b' + str(l)]
            Z = np.dot(W, A) + b
            if l != self.layers - 1:
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

    def predict(self, X):
        """predict(x,params)
        It will use trainned model to predict the result, if use want to predict
        with you own model, you can call set_pramas(parameters) at firt,
        and than call this function.
        x: the input of test data
        y: the output of test data
        """
        res,_ = self.forward(X)
        return res


    def backward(self, X, y, params=None):
        """
        Backward propagation of neural network
        Parameters
        ----------
        X : np.ndarray (features, samples)
            the input of test data
        y : np.ndarray (classes, samples)
            the output of test data
        params : python dictionary (if it is not set, the function will use self.params instead)
            python dictionary containing your parameters "Wl", "bl"(l means the lth layer) 
        Returns
        -------
        gradients: python dictionary
            A dictionary with the gradients of each parameter
        """
        if params is not None:
            self.params = params
        Al,cache = self.forward(X)
        cost = compute_cost(self.params,self.hyperparams,y,Al)
        m = X.shape[1]
        # activation function
        activation = self.hyperparams['activation']
        L2_penalty = self.hyperparams['L2_penalty']
        gradients = {}

        for l in range(self.layers-1,0,-1):
            Z = cache['Z'+str(l)]
            A = cache['A'+str(l-1)]
            if l == self.layers-1 :
                dZ = Al - y
                dW = 1./m * np.dot(dZ, A.T)
                db = 1./m * np.sum(dZ, axis=1, keepdims = True)
            else:
                # use W in previous layer
                Wp = self.params['W'+str(l+1)]
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
            dW += 1.0*L2_penalty/m * self.params['W'+str(l)]

            gradients['dZ'+str(l)] = dZ
            gradients['dW'+str(l)] = dW
            gradients['db'+str(l)] = db

        return gradients, cost

    def update_no_improve_count(self, X_val=None, y_val=None):
        if self.hyperparams['early_stopping']:
            # compute validation score, use that for stopping
            self.validation_scores.append(self.score(X_val, y_val))

            print("Validation score: %f" % self.validation_scores[-1])
            # update best parameters
            # use validation_scores, not loss_curve_
            # let's hope no-one overloads .score with mse
            last_valid_score = self.validation_scores[-1]

            if last_valid_score < (self.best_validation_score + self.hyperparams['tol']):
                self.no_improve_count += 1
            else:
                self.no_improve_count = 0

            if last_valid_score > self.best_validation_score:
                self.best_validation_score = last_valid_score
        else:
            if self.costs[-1] > self.best_cost - self.hyperparams['tol']:
                self.no_improve_count += 1
            else:
                self.no_improve_count = 0
            if self.costs[-1] < self.best_cost:
                self.best_cost = self.costs[-1]

    def trigger_stopping(self):
        if self.iters_count >= self.hyperparams['max_iters']:
            msg = ("Reached the max iterations %d, but training loss "
                   "improved less than tol=%f." % (
                       self.hyperparams['max_iters'], self.hyperparams['tol']))
            print(msg)
            return True

        if self.no_improve_count > self.hyperparams['no_improve_num']:
            # not better than last `no_improve_count` iterations by tol
            # stop or decrease learning rate
            if self.hyperparams['early_stopping']:
                msg = ("Validation score did not improve more than "
                       "tol=%f for %d consecutive epochs after %d iterations." % (
                           self.hyperparams['tol'], self.no_improve_count, self.iters_count))
            else:
                msg = ("Training loss did not improve more than tol=%f"
                       " for %d consecutive epochs after %d iterations." % (
                           self.hyperparams['tol'], self.no_improve_count, self.iters_count))
            print(msg)
            return True

        return False

    def algorithm_init(self):
        """
        Initialize some parameters before start algorithm.
        """
        self.iters_count = 0
        self.no_improve_count = 0
        # cost values of on every updated parameters
        self.costs = []
        # the best(minimal) cost value of all the current cost values
        self.best_cost = 1e10
        self.validation_scores = []
        self.best_validation_score = 0.0
        units = self.hyperparams['units']
        self.layers = len(units)
        self.params = init_params(units)

    def BGD(self, X, y):
        """
        Batch gradient descent
        """
        self.algorithm_init()
        # the number of iterations
        max_iters = self.hyperparams['max_iters']
        learning_rate_init = self.hyperparams['learning_rate_init']

        for self.iters_count in range(1,self.hyperparams['max_iters']+1):
            grads, cost = self.backward(X,y)
            self.costs.append(cost)
            # update parameters with calculated gradients
            for l in range(1,self.layers):
                self.params['W' + str(l)] -= learning_rate_init * grads['dW' + str(l)]
                self.params['b' + str(l)] -= learning_rate_init * grads['db' + str(l)]
            self.update_no_improve_count()
            if self.trigger_stopping():
                break

        return self.params

    def MBGD(self, X, y):
        """
        Mini batch gradient descent
        """
        self.algorithm_init()
        num = X.shape[1]
        batch_size = self.hyperparams['batch_size']
        learning_rate_init = self.hyperparams['learning_rate_init']

        for self.iters_count in range(1,self.hyperparams['max_iters']+1):
            costs_sum = 0.0
            for batch_slice in gen_batches(num, batch_size):
                grads, cost = self.backward(X[:,batch_slice],y[:,batch_slice])
                costs_sum += cost * (batch_slice.stop - batch_slice.start)
                # update parameters with calculated gradients
                for l in range(1,self.layers):
                    self.params['W' + str(l)] -= learning_rate_init * grads['dW' + str(l)]
                    self.params['b' + str(l)] -= learning_rate_init * grads['db' + str(l)]
            self.costs.append(costs_sum / X.shape[1])
            self.update_no_improve_count()
            if self.trigger_stopping():
                break

        return self.params

    SOLVERS = {'BGD':BGD, 'MBGD':MBGD}


    def fit(self, X, y):
        """Fit the model to data matrix X and target(s) y.
        Parameters
        ----------
        X : array-like or sparse matrix, shape (n_samples, n_features)
            The input data.
        y : array-like, shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels in classification, real numbers in
            regression).
        Returns
        -------
        params : returns a trained model.
        """
        solver_name = self.hyperparams['solver']
        solver = self.SOLVERS[solver_name]
        self.params = solver(self,X,y)
        return self.params
