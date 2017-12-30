import os
import sys
sys.path.append('..')
import time
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
    batch_size : int, default '64'
        Size of minibatches for minibatch gradient descent.
        If the solver is 'BGD', the classifier will not use minibatch.
    no_improve_num : int, optional, default 10
        Maximum number of epochs to not meet ``tol`` improvement.
        Only effective when solver='sgd' or 'adam'
    early_stopping : bool, default False
        Whether to use early stopping to terminate training when validation
        score is not improving. If set to true, it will automatically set
        aside 10% of training data as validation and terminate training when
        validation score is not improving by at least tol for
        ``no_improve_num`` consecutive epochs.
        Only effective when solver='MBGD' or 'ADAM'
    momentum_beta : float, optional, default 0.9
        The momentum hyperparameter for Momentum/NAG/Adam algorithms.
        The larger the momentum β is, the smoother
        the update because the more we take the past gradients into account.
        But if β is too big, it could also smooth out the updates too much.
        Common values for β range from 0.8 to 0.999. If you don't feel inclined
        to tune this, β=0.9 is often a reasonable default.
    rms_beta : float, optional, default 0.999
        The hyperparameter for RMSprop/Adam algorithms.
    epsilon : float, optional, default 1e-8
        Value for numerical stability in RMSprop/Adam.
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
                 verbose=False, no_improve_num=10, early_stopping=False,
                 momentum_beta=0.9, rms_beta=0.99, epsilon=1e-8):

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
        self.hyperparams['verbose'] = verbose
        self.hyperparams['no_improve_num'] = no_improve_num
        self.hyperparams['early_stopping'] = early_stopping
        self.hyperparams['momentum_beta'] = momentum_beta
        self.hyperparams['rms_beta'] = rms_beta
        self.hyperparams['epsilon'] = epsilon

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
        out_activation = self.hyperparams['out_activation']
        # cache[0] is cached Z of each layer
        # cache[1] is cached A of each layer
        cache = 2*[np.array(self.layers*[None])]
        cache = np.array(cache)

        cache[0,0] = X
        cache[1,0] = X

        A = X
        for l in range(1, self.layers):
            W = self.params[0,l]
            b = self.params[1,l]
            Z = np.dot(W, A) + b
            # the media layer
            if l != self.layers - 1:
                if activation == 'relu':
                    A = relu(Z)
                elif activation == 'sigmoid':
                    A = sigmoid(Z)
                else:
                    raise ValueError('No such activation: %s' % (activation))
		    # the last layer is using sigmoid to do classification
            else:
                if out_activation == 'softmax':
                    A = softmax(Z)
                elif out_activation == 'sigmoid':
                    A = sigmoid(Z)
                else:
                    raise ValueError('No such out activation: %s' % (activation))
            cache[0,l] = Z
            cache[1,l] = A

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
        out_activation = self.hyperparams['out_activation']
        L2_penalty = self.hyperparams['L2_penalty']
        gradients = 2*[np.array(self.layers*[None])]
        gradients = np.array(gradients)
        gradients[0,0] = gradients[1,0] = np.array([])
        for l in range(self.layers-1,0,-1):
            Z = cache[0,l]
            A = cache[1,l-1]
            if l == self.layers-1 :
                dZ = Al - y
                dW = 1./m * np.dot(dZ, A.T)
                db = 1./m * np.sum(dZ, axis=1, keepdims = True)
            else:
                # use W in previous layer
                Wp = self.params[0,l+1]
                # use calculated dZ calculate in the previous iteration
                dA = np.dot(Wp.T, dZ)
                if activation == 'relu':
                    # np.int64(A > 0) is the gradient of ReLU
                    dZ = dA * np.int64(Z > 0)
                elif activation == 'sigmoid':
                    gZ = sigmoid(Z)
                    dZ = dA * gZ * (1-gZ)
                else:
                    raise ValueError('No such activation: %s' % (activation))

                dW = 1./m * np.dot(dZ, A.T)
                db = 1./m * np.sum(dZ, axis=1, keepdims = True)

            # add regularition item
            dW += 1.0*L2_penalty/m * self.params[0,l]

            gradients[0,l] = dW
            gradients[1,l] = db

        return gradients, cost

    def update_no_improve_count(self, X_val=None, y_val=None):
        if self.hyperparams['early_stopping']:
            # compute validation score, use that for stopping
            self.validation_scores.append(self.score(X_val, y_val))
            if self.hyperparams['verbose']:
                print("Validation score: %f" % (self.validation_scores[-1]) )
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
            if self.hyperparams['verbose']:
                print("Cost value: %f" % (self.costs[-1]) )
            if self.costs[-1] > self.best_cost - self.hyperparams['tol']:
                self.no_improve_count += 1
            else:
                self.no_improve_count = 0
            if self.costs[-1] < self.best_cost:
                self.best_cost = self.costs[-1]

    def trigger_stopping(self):
        if self.iters_count >= self.hyperparams['max_iters']:
            msg = ("Reached the max iterations %d, but training loss "
                   "improved still more than tol=%f." % (
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
            print(self.params[0,0].shape,grads[0,0].shape)
            # update parameters with calculated gradients
            self.params -= learning_rate_init * grads
            self.update_no_improve_count()
            if self.trigger_stopping():
                break

        return self.params

    def Momentum(self, X, y):
        """
        Momentum gradient gradient descent
          Because mini-batch gradient descent makes a parameter update
        after seeing just a subset of examples, the direction of the
        update has some variance, and so the path taken by mini-batch
        gradient descent will "oscillate" toward convergence. Using
        momentum can reduce these oscillations.
          Momentum takes into account the past gradients to smooth out
        the update. We will store the 'direction' of the previous gradients
        in the variable vv. Formally, this will be the exponentially
        weighted average of the gradient on previous steps. You can also
        think of vv as the "velocity" of a ball rolling downhill,
        building up speed (and momentum) according to the direction of
        the gradient/slope of the hill.
        """
        self.algorithm_init()
        X_test = None
        y_test = None
        if self.hyperparams['early_stopping']:
            X, y, X_test, y_test = train_test_split(X,y)
        num = X.shape[1]
        batch_size = self.hyperparams['batch_size']
        learning_rate_init = self.hyperparams['learning_rate_init']

        units = self.hyperparams['units']

        momentumW = self.layers * [None]
        momentumb = self.layers * [None]
        for l in range(1,self.layers):
            momentumW[l] = np.zeros((units[l], units[l-1]))
            momentumb[l] = np.zeros((units[l], 1))

        moment_beta = self.hyperparams['momentum_beta']
        for self.iters_count in range(1,self.hyperparams['max_iters']+1):
            costs_sum = 0.0
            for batch_slice in gen_batches(num, batch_size):
                grads, cost = self.backward(X[:,batch_slice],y[:,batch_slice])
                costs_sum += cost * (batch_slice.stop - batch_slice.start)
                # update parameters with calculated gradients
                for l in range(1,self.layers):
                    # These two methods to update momentum are correct, they work well on different learning rate
                    #momentumW[l] = moment_beta*momentumW[l] + (1-moment_beta)*grads['dW' + str(l)]
                    #momentumb[l] = moment_beta*momentumb[l] + (1-moment_beta)*grads['db' + str(l)]
                    momentumW[l] = moment_beta*momentumW[l] + grads['dW' + str(l)]
                    momentumb[l] = moment_beta*momentumb[l] + grads['db' + str(l)]
                    self.params[l] -= learning_rate_init * momentumW[l]
                    self.params[l] -= learning_rate_init * momentumb[l]
            self.costs.append(costs_sum / X.shape[1])
            self.update_no_improve_count(X_test, y_test)
            if self.trigger_stopping():
                break

        return self.params

    def NAG(self, X, y):
        """
        Nesterov accelerated gradient
        In this algorithm you'd better turn on the early_stopping
        for the cost valueis not the cost value we need
        """
        self.algorithm_init()
        X_test = None
        y_test = None
        if self.hyperparams['early_stopping']:
            X, y, X_test, y_test = train_test_split(X,y)
        num = X.shape[1]
        batch_size = self.hyperparams['batch_size']
        learning_rate_init = self.hyperparams['learning_rate_init']

        units = self.hyperparams['units']

        momentumW = self.layers * [None]
        momentumb = self.layers * [None]
        for l in range(1,self.layers):
            momentumW[l] = np.zeros((units[l], units[l-1]))
            momentumb[l] = np.zeros((units[l], 1))

        moment_beta = self.hyperparams['momentum_beta']
        for self.iters_count in range(1,self.hyperparams['max_iters']+1):
            costs_sum = 0.0
            for batch_slice in gen_batches(num, batch_size):
                for l in range(1,self.layers):
                    self.params[l] -= learning_rate_init * moment_beta * momentumW[l]
                    self.params[l] -= learning_rate_init * moment_beta * momentumb[l]
                grads, cost = self.backward(X[:,batch_slice],y[:,batch_slice])
                costs_sum += cost * (batch_slice.stop - batch_slice.start)
                # update parameters with calculated gradients
                for l in range(1,self.layers):
                    momentumW[l] = moment_beta*momentumW[l] + grads['dW' + str(l)]
                    momentumb[l] = moment_beta*momentumb[l] + grads['db' + str(l)]
                    self.params[l] -= learning_rate_init * momentumW[l]
                    self.params[l] -= learning_rate_init * momentumb[l]
            self.costs.append(costs_sum / X.shape[1])
            self.update_no_improve_count(X_test, y_test)
            if self.trigger_stopping():
                break

        return self.params

    def RMSprop(self, X, y):
        """
        RMSprop algorithm
        """
        self.algorithm_init()
        X_test = None
        y_test = None
        if self.hyperparams['early_stopping']:
            X, y, X_test, y_test = train_test_split(X,y)
        num = X.shape[1]
        batch_size = self.hyperparams['batch_size']
        learning_rate_init = self.hyperparams['learning_rate_init']

        units = self.hyperparams['units']

        sW = self.layers * [None]
        sb = self.layers * [None]
        for l in range(1,self.layers):
            sW[l] = np.zeros((units[l], units[l-1]))
            sb[l] = np.zeros((units[l], 1))
        rms_beta = self.hyperparams['rms_beta']
        epsilon = self.hyperparams['epsilon']
        for self.iters_count in range(1,self.hyperparams['max_iters']+1):
            costs_sum = 0.0
            for batch_slice in gen_batches(num, batch_size):
                grads, cost = self.backward(X[:,batch_slice],y[:,batch_slice])
                costs_sum += cost * (batch_slice.stop - batch_slice.start)
                # update parameters with calculated gradients
                for l in range(1,self.layers):
                    sW[l] = rms_beta*sW[l] + (1-rms_beta)*grads['dW' + str(l)]**2
                    sb[l] = rms_beta*sb[l] + (1-rms_beta)*grads['db' + str(l)]**2
                    # bias correct
                    corsW = sW[l] / (1. - np.power(rms_beta,self.iters_count))
                    corsb = sb[l] / (1. - np.power(rms_beta,self.iters_count))
                    self.params[l] -= learning_rate_init * grads['dW' + str(l)]/np.sqrt(corsW+epsilon)
                    self.params[l] -= learning_rate_init * grads['db' + str(l)]/np.sqrt(corsb+epsilon)
            self.costs.append(costs_sum / X.shape[1])
            self.update_no_improve_count(X_test, y_test)
            if self.trigger_stopping():
                break

        return self.params

    def Adam(self, X, y):
        """
        Adam algorithm
        """
        self.algorithm_init()
        X_test = None
        y_test = None
        if self.hyperparams['early_stopping']:
            X, y, X_test, y_test = train_test_split(X,y)
        num = X.shape[1]
        batch_size = self.hyperparams['batch_size']
        learning_rate_init = self.hyperparams['learning_rate_init']

        units = self.hyperparams['units']

        momentumW = self.layers * [None]
        momentumb = self.layers * [None]
        sW = self.layers * [None]
        sb = self.layers * [None]

        for l in range(1,self.layers):
            momentumW[l] = np.zeros((units[l], units[l-1]))
            momentumb[l] = np.zeros((units[l], 1))
            sW[l] = np.zeros((units[l], units[l-1]))
            sb[l] = np.zeros((units[l], 1))

        rms_beta = self.hyperparams['rms_beta']
        moment_beta = self.hyperparams['momentum_beta']
        epsilon = self.hyperparams['epsilon']
        for self.iters_count in range(1,self.hyperparams['max_iters']+1):
            costs_sum = 0.0
            for batch_slice in gen_batches(num, batch_size):
                grads, cost = self.backward(X[:,batch_slice],y[:,batch_slice])
                costs_sum += cost * (batch_slice.stop - batch_slice.start)
                for l in range(1,self.layers):
                    momentumW[l] = moment_beta*momentumW[l] + (1-moment_beta)*grads['dW' + str(l)]
                    momentumb[l] = moment_beta*momentumb[l] + (1-moment_beta)*grads['db' + str(l)]
                    # bias correct
                    cormW = momentumW[l] / (1. - np.power(moment_beta,self.iters_count))
                    cormb = momentumb[l] / (1. - np.power(moment_beta,self.iters_count))
                    sW[l] = rms_beta*sW[l] + (1-rms_beta)*grads['dW' + str(l)]**2
                    sb[l] = rms_beta*sb[l] + (1-rms_beta)*grads['db' + str(l)]**2
                    # bias correct
                    corsW = sW[l] / (1. - np.power(rms_beta,self.iters_count))
                    corsb = sb[l] / (1. - np.power(rms_beta,self.iters_count))
                    # update parameters
                    self.params[l] -= learning_rate_init * cormW/np.sqrt(corsW+epsilon)
                    self.params[l] -= learning_rate_init * cormb/np.sqrt(corsb+epsilon)
            self.costs.append(costs_sum / X.shape[1])
            self.update_no_improve_count(X_test, y_test)
            if self.trigger_stopping():
                break

        return self.params


    SOLVERS = {'BGD':BGD, 'MBGD':MBGD, 'Momentum':Momentum, 'NAG':NAG, 'RMSprop':RMSprop, 'Adam':Adam }


    def fit(self, X, y):
        """Fit the model to data matrix X and target(s) y.
        Parameters
        ----------
        X : array-like or sparse matrix, shape (n_features, n_samples)
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
        tic = time.time()
        self.params = solver(self,X,y)
        toc = time.time()
        print ("-------- Computation time  on dataset with algorithm %s is %f ms" % (solver_name, toc - tic))

        return self.params
