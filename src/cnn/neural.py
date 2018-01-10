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
from utils import *
from loss import compute_cnn_cost
from base import Classifier


class LeNet5(Classifier):
    """
    classes : tuple, list, np.ndarray
        This constains the different classes of you labels. len(classes) should
        be equal to the last value of units(except when it is 1 meaning binary classification )
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
    conv_filter_size : int, default is 5
        the size of convolution filters, all the filter sizes of convolution layers are the same 
    conv_pad : int, default is 0(no padding)
        mount of padding around each image on vertical and horizontal dimensions
    conv_stride : int. default 1
        the stride of convolution
    pool_filter_size : int, default 2
        the size of pool filter
    pool_stride : int, default 2
        the stride of pool
    pool_mode : string ("max" or "average"), default "max"
        the pooling mode you would like to use
    """

    hyperparams = {}
    params = {}
    # how many layers of the neural network
    fc_layers = 0
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

    def __init__(self, model_type='classification',solver='BGD',
                 lossfunc='softmax_loss',learning_rate_type='constant',
                 learning_rate_init=0.01,
                 L2_penalty=0.,max_iters=200, batch_size=64, tol=1e-4,
                 verbose=False, no_improve_num=10, early_stopping=False,
                 momentum_beta=0.9, rms_beta=0.99, epsilon=1e-8,
                 conv_filter_size=5, conv_pad=0, conv_stride=1,
                 pool_filter_size=2, pool_stride=2, pool_mode="max"):

        self.hyperparams['solver'] = solver
        self.hyperparams['model_type'] = model_type
        self.hyperparams['lossfunc'] = lossfunc
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
        self.hyperparams['conv_filter_size'] = conv_filter_size
        self.hyperparams['conv_pad'] = conv_pad
        self.hyperparams['conv_stride'] = conv_stride
        self.hyperparams['pool_filter_size'] = pool_filter_size
        self.hyperparams['pool_stride'] = pool_stride
        self.hyperparams['pool_mode'] = pool_mode

        self.layers = load_config_layers()
        self.hyperparams['layers'] = self.layers

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
        self.layers = hyperparams['layers']

    def forward(self, X, params=None):
        """Forward propatation of neural network
        Parameters
        ----------
        X : np.ndarray (samples, features)
            the input of test data
        params : python dictionary (if it is not set, the function will use self.params instead)
            python dictionary containing your parameters "Wl", "bl"(l means the lth layer)
        Returns
        -------
        A : the result of last layer
        cache : "Wl", "bl"(l means the lth layer)of all the layers
        """
        if params is not None:
            self.params = params
        caches = []
        caches.append(None)
        A_in = X
        first_fc = True
        keys = sorted(self.layers.keys())
        for l in range(1, len(self.layers)):
            layer = self.layers[keys[l]]
            W = self.params[0, l]
            b = self.params[1, l]
            if layer['layer_type'] == 'conv':
                A_out, cache = conv_forward(A_in, W, b, layer)
            if layer['layer_type'] == 'pool':
                A_out, cache = pool_forward(A_in, W, b, layer)
            if layer['layer_type'] == 'fc':
                A_in = A_in.reshape(A_in.shape[0], -1)
                A_out, cache = fc_forward(A_in, W, b, layer)
            caches.append(cache)
            # the output of this layer is the next layer's input
            A_in = A_out

        return A_out, caches

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
        X : np.ndarray (samples, features)
            the input of test data
        y : np.ndarray (samples, classes)
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
        layer_num = len(self.layers)
        m, n = y.shape
        A_out, caches = self.forward(X)
        # activation function
        L2_penalty = self.hyperparams['L2_penalty']
        gradients = init_empty(2,len(self.layers))
        cost = compute_cnn_cost(self.hyperparams,y,A_out)
        keys = sorted(self.layers.keys())
        for l in range(layer_num - 1, 0, -1):
            Z, A_in, W, b, layer = caches[l]

            # the last layer
            if l == layer_num - 1:
                dZ = 1./m * (A_out - y)
            elif 'activation' in layer.keys():
                activation = layer['activation']
                if activation == 'relu':
                    # np.int64(A > 0) is the gradient of ReLU
                    dZ = dA_in * np.int64(Z > 0)
                elif activation == 'sigmoid':
                    gZ = sigmoid(Z)
                    dZ = dA_in * gZ * (1-gZ)
                else:
                    raise ValueError('No such activation: %s' % (activation))
            else:
                # if there is no activation function
                dZ = dA_in

            if layer['layer_type'] == 'conv':
                dZ = dZ.reshape(layer['shape'])
                dA_in, dW, db = conv_backward(dZ, caches[l])
                # add regularition item
                dW += 1.0*L2_penalty/m * W
                print('conv')
            if layer['layer_type'] == 'pool':
                dZ = dZ.reshape(layer['shape'])
                dA_in, dW, db = pool_backward(dZ, caches[l])
                print('pool')
            if layer['layer_type'] == 'fc':
                dZ = dZ.reshape(layer['shape'])
                dA_in, dW, db = fc_backward(dZ, caches[l])
                # add regularition item
                dW += 1.0*L2_penalty/m * W
                print('fc')

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
        self.params = init_cnn_params(self.layers)

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
            self.params -= learning_rate_init * grads
            self.update_no_improve_count()
            if self.trigger_stopping():
                break

        return self.params


    SOLVERS = {'BGD':BGD }


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
        #forward()
        """
        solver_name = self.hyperparams['solver']
        solver = self.SOLVERS[solver_name]
        tic = time.time()
        self.params = solver(self,X,y)
        toc = time.time()
        print ("-------- Computation time  on dataset with algorithm %s is %f ms" % (solver_name, toc - tic))

        return self.params

        """
