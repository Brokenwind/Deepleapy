import sys
sys.path.append('..')
import numpy as np
from datamap import *
from prodata import *
from loss import compute_cnn_cost
from neural import LeNet5
from utils import *

class GradientCheck:
    deadline = 1e-8
    network = None
    params = {}
    hyperparams = {}
    X = None
    y = None
    test_num = 5
    layers = None
    check_step = 1e-4

    def __init__(self):
        self.network = LeNet5()
        self.hyperparams = self.network.get_hyperparams()
        self.layers = load_config_layers('layers_check_2.json')
        self.hyperparams['layers'] = self.layers
        L = len(self.layers)
        # the number of ouput classifications
        first_layer = self.layers['layer0']
        last_layer = self.layers['layer'+str(L-1)]
        class_num = last_layer['fc_units'][1]
        data_size = first_layer['data_size']
        map = DataMap(range(0,class_num))

        self.params = init_cnn_params(self.layers)
        self.X = np.random.uniform(size=data_size)
        y = np.random.uniform(0,class_num,size=(data_size[0],)).astype(int)
        self.y = map.class2matrix(y).T
        print(self.X.shape)
        print(self.y.shape)

    def checks(self):
        self.check_sigmoid()
        self.check_softmax()

    def check_sigmoid(self):
        #self.hyperparams['L2_penalty'] = 0.1
        # check sigmoid relative
        self.hyperparams['lossfunc'] = 'log_loss'
        #self.hyperparams['out_activation'] = 'sigmoid'
        self.network.set_hyperparams(self.hyperparams)
        self.check_gradient()

    def check_softmax(self):
        #self.hyperparams['L2_penalty'] = 0.1
        # check softmax relative
        self.hyperparams['lossfunc'] = 'softmax_loss'
        #self.hyperparams['out_activation'] = 'softmax'
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
        layers = self.hyperparams['layers']
        grads=init_empty(2,len(layers))
        for prefix in range(0,2):
            for l in range(1,len(layers)):
                value = self.params[prefix,l]
                layer = layers['layer'+str(l)]
                if layer['layer_type'] == 'conv':
                    dTmp = self.update_conv_params(value)
                elif layer['layer_type'] == 'pool':
                    dTmp = np.array([])
                elif layer['layer_type'] == 'fc':
                    dTmp = self.update_fc_params(value)
                grads[prefix,l] = dTmp
        return grads

    def update_conv_params(self, value):
        f1,f2,inc,outc = value.shape
        dTmp = np.zeros(value.shape)
        for i in range(0,f1):
            for j in range(0,f2):
                for k in range(0,inc):
                    for l in range(0,outc):
                        tmp = value[i,j,k,l]
                        value[i,j,k,l] = tmp + self.check_step
                        y_prob,_ = self.network.forward(self.X, params=self.params)
                        up = compute_cnn_cost(self.hyperparams, self.y, y_prob)
                        value[i,j,k,l] = tmp - self.check_step
                        self.network.set_params(self.params)
                        y_prob,_ = self.network.forward(self.X, params=self.params)
                        down = compute_cnn_cost(self.hyperparams, self.y, y_prob)
                        dTmp[i,j,k,l] = (up - down)/(2.0*self.check_step)
                        # reset to the original value
                value[i,j] = tmp

        return dTmp

    def update_fc_params(self, value):
        m,n = value.shape
        dTmp = np.zeros((m,n))
        for i in range(0,m):
            for j in range(0,n):
                tmp = value[i,j]
                value[i,j] = tmp + self.check_step
                y_prob,_ = self.network.forward(self.X, params=self.params)
                up = compute_cnn_cost(self.hyperparams, self.y, y_prob)
                value[i,j] = tmp - self.check_step
                self.network.set_params(self.params)
                y_prob,_ = self.network.forward(self.X, params=self.params)
                down = compute_cnn_cost(self.hyperparams, self.y, y_prob)
                dTmp[i,j] = (up - down)/(2.0*self.check_step)
                # reset to the original value
                value[i,j] = tmp

        return dTmp

    def check_gradient(self):
        """
        Checks if backward propagation computes correctly the gradient of the cost function

    Arguments:
        self.hyperparams -- python dictionary containing your hyperparameters
        self.test_num -- how many test samples you will use
        Returns:
        difference -- difference between the approximated gradient and the backward propagation gradient

        """
        #print('\nChecking %s ...' % (self.hyperparams['out_activation']))
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
