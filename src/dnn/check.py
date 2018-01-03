import sys
sys.path.append('..')
import numpy as np
from datamap import *
from prodata import *
from loss import compute_cost

class GradientCheck:

    deadline = 1e-8
    network = None
    params = {}
    hyperparams = {}
    X = None
    y = None
    test_num = 5

    def __init__(self,network,hyperparams=None):
        self.network = network
        units = [4,5,3]
        self.hyperparams = self.network.get_hyperparams()
        self.hyperparams['units'] = units
        L = len(units)
        # the number of ouput classifications
        class_num = units[L-1]
        # the number of features of input data
        feature_num = units[0]
        map = DataMap(range(0,class_num))

        self.params = debug_init_params(units)

        # Reusing debug_init_self.paramsializeWeights to generate X
        self.X,_ = debug_init_unit( self.test_num, feature_num )
        # generate corresponding y
        y = np.mod(np.arange(1,self.test_num+1),class_num)
        self.y = map.class2matrix(y)

    def checks(self):
        self.check_sigmoid()
        self.check_softmax()

    def check_sigmoid(self):
        self.hyperparams['L2_penalty'] = 0.1
        # check sigmoid relative
        self.hyperparams['lossfunc'] = 'log_loss'
        self.hyperparams['out_activation'] = 'sigmoid'
        self.network.set_hyperparams(self.hyperparams)
        self.check_gradient()

    def check_softmax(self):
        self.hyperparams['L2_penalty'] = 0.1
        # check softmax relative
        self.hyperparams['lossfunc'] = 'softmax_loss'
        self.hyperparams['out_activation'] = 'softmax'
        self.network.set_hyperparams(self.hyperparams)
        self.check_gradient()

    def numerical_gradient(self):
        """
        Compute all the numerical gradients of parameters in cost function
        Arguments:
        self.params -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3":
        self.hyperparams -- python dictionary containing your hyperparameters

        X -- input datapoint, of shape (input size, 1)
        y -- true "label"

    Returns:
        grads -- numerical gradients of the cost function

        """
        check = 1e-4
        units = self.hyperparams['units']
        grads=2*[np.array(len(units)*[None])]
        grads = np.array(grads)
        for prefix in range(0,2):
            for l in range(1,len(units)):
                value = self.params[prefix,l]
                m,n = value.shape
                dTmp = np.zeros((m,n))
                for i in range(0,m):
                    for j in range(0,n):
                        tmp = value[i,j]
                        value[i,j] = tmp + check
                        y_prob,_ = self.network.forward(self.X, params=self.params)
                        up = compute_cost(self.params, self.hyperparams, self.y, y_prob)
                        value[i,j] = tmp - check
                        self.network.set_params(self.params)
                        y_prob,_ = self.network.forward(self.X, params=self.params)
                        down = compute_cost(self.params, self.hyperparams, self.y, y_prob)
                        dTmp[i,j] = (up - down)/(2.0*check)
                        # reset to the original value
                        value[i,j] = tmp
                grads[prefix,l] = dTmp
        return grads

    def check_gradient(self):
        """
        Checks if backward propagation computes correctly the gradient of the cost function

    Arguments:
        self.hyperparams -- python dictionary containing your hyperparameters
        self.test_num -- how many test samples you will use
        Returns:
        difference -- difference between the approximated gradient and the backward propagation gradient

        """
        print('\nChecking %s ...' % (self.hyperparams['out_activation']))
        # calculate the gradient with two diffent ways
        grad1,_ = self.network.backward(self.X, self.y, params=self.params)
        grad2 = self.numerical_gradient()

        # calculate the norm of the difference of two kinds of W
        diff = normdiff(grad1,grad2)
        print ("Evaluate the norm of the difference between two dW: %e" % (diff[0]))
        # calculate the norm of the difference of two kinds of b
        print ("Evaluate the norm of the difference between two db: %e" % (diff[1]))

        res = ( diff[0] < self.deadline ) and ( diff[1] < self.deadline )
        if ( res ):
            print ("Passed the gradient check!")
        else:
            print ("Did not passed the gradient check!")

        return res
