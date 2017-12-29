import sys
import time
sys.path.append('../')
sys.path.append('../../')
from neural import *
from datamap import *
from loss import *
from check import GradientCheck
from score import accuracy_score

def load_params(fpath):
    params = {}
    data = np.load(fpath)
    params['b1'] = data['b1']
    params['W1'] = data['W1']
    params['b2'] = data['b2']
    params['W2'] = data['W2']

    return params

def fitBGD(network,X,Y):
    hyperparams = {}
    units = [400, 25, 10]
    hyperparams['solver'] = 'BGD'
    hyperparams['units'] = units
    hyperparams['lossfunc'] = 'log_loss'
    hyperparams['activation'] = 'relu'
    hyperparams['out_activation'] = 'sigmoid'
    hyperparams['L2_penalty'] = 1.0
    hyperparams['tol'] = 0.001
    hyperparams['max_iters'] = 500
    hyperparams['learning_rate_init'] = 0.2

    network.set_hyperparams(hyperparams)
    params = network.fit(X,Y)
    # test forward propagation
    accuracy = network.score(X,Y)
    print ('The accuracy on hand-written digit dataset is: {}%'.format( accuracy * 100 ))


def fitMBGD(network,X,Y):
    hyperparams = {}
    units = [400, 25, 10]
    hyperparams['solver'] = 'MBGD'
    hyperparams['units'] = units
    hyperparams['activation'] = 'relu'
    hyperparams['out_activation'] = 'sigmoid'
    hyperparams['lossfunc'] = 'log_loss'
    hyperparams['L2_penalty'] = 1.0
    hyperparams['max_iters'] = 200
    hyperparams['tol'] = 0.0001
    hyperparams['learning_rate_init'] = 0.15
    hyperparams['batch_size'] = 512
    #hyperparams['early_stopping'] = True

    network.set_hyperparams(hyperparams)
    params = network.fit(X,Y)
    # test forward propagation
    accuracy = network.score(X,Y)
    print ('The accuracy on hand-written digit dataset is: {}%'.format( accuracy * 100 ))


def fitMBGD2(network,X,Y):
    hyperparams = {}
    units = [400, 25, 10]
    hyperparams['solver'] = 'MBGD'
    hyperparams['units'] = units
    hyperparams['activation'] = 'relu'
    hyperparams['out_activation'] = 'softmax'
    hyperparams['lossfunc'] = 'softmax_loss'
    hyperparams['L2_penalty'] = 1.0
    hyperparams['max_iters'] = 200
    hyperparams['tol'] = 0.001
    hyperparams['learning_rate_init'] = 0.15
    hyperparams['batch_size'] = 512
    hyperparams['early_stopping'] = True

    network.set_hyperparams(hyperparams)
    params = network.fit(X,Y)
    # test forward propagation
    accuracy = network.score(X,Y)
    print ('The accuracy on hand-written digit dataset is: {}%'.format( accuracy * 100 ))

def fitMomentum(network,X,Y):
    hyperparams = {}
    units = [400, 25, 10]
    hyperparams['solver'] = 'Momentum'
    hyperparams['units'] = units
    hyperparams['activation'] = 'relu'
    hyperparams['out_activation'] = 'softmax'
    hyperparams['lossfunc'] = 'softmax_loss'
    hyperparams['L2_penalty'] = 1.0
    hyperparams['max_iters'] = 200
    hyperparams['tol'] = 0.0001
    hyperparams['learning_rate_init'] = 0.15
    hyperparams['batch_size'] = 512
    hyperparams['early_stopping'] = True
    hyperparams['verbose'] = False

    network.set_hyperparams(hyperparams)
    params = network.fit(X,Y)

    # test forward propagation
    accuracy = network.score(X,Y)
    print ('The accuracy on hand-written digit dataset is: {}%'.format( accuracy * 100 ))

def fitNAG(network,X,Y):
    hyperparams = {}
    units = [400, 25, 10]
    hyperparams['solver'] = 'NAG'
    hyperparams['units'] = units
    hyperparams['activation'] = 'relu'
    hyperparams['out_activation'] = 'softmax'
    hyperparams['lossfunc'] = 'softmax_loss'
    hyperparams['L2_penalty'] = 1.0
    hyperparams['max_iters'] = 200
    hyperparams['tol'] = 0.0001
    hyperparams['learning_rate_init'] = 0.15
    hyperparams['batch_size'] = 512
    hyperparams['early_stopping'] = True
    #hyperparams['verbose'] = False

    network.set_hyperparams(hyperparams)
    params = network.fit(X,Y)

    # test forward propagation
    accuracy = network.score(X,Y)
    print ('The accuracy on hand-written digit dataset is: {}%'.format( accuracy * 100 ))

def fitRMSprop(network,X,Y):
    hyperparams = {}
    units = [400, 25, 10]
    hyperparams['solver'] = 'RMSprop'
    hyperparams['units'] = units
    hyperparams['activation'] = 'relu'
    hyperparams['out_activation'] = 'softmax'
    hyperparams['lossfunc'] = 'softmax_loss'
    hyperparams['L2_penalty'] = 1.0
    hyperparams['max_iters'] = 200
    hyperparams['tol'] = 0.0001
    hyperparams['learning_rate_init'] = 0.01
    hyperparams['batch_size'] = 512
    hyperparams['early_stopping'] = True
    #hyperparams['verbose'] = True

    network.set_hyperparams(hyperparams)
    params = network.fit(X,Y)

    # test forward propagation
    accuracy = network.score(X,Y)
    print ('The accuracy on hand-written digit dataset is: {}%'.format( accuracy * 100 ))


if __name__ == '__main__':
    fpath = '../../../dataset/hand_written_digit.npz'
    map = DataMap([1,2,3,4,5,6,7,8,9,0])
    data = np.load(fpath)
    params = load_params(fpath)

    X = data['X']
    y = data['y']

    Y = map.class2matrix(y)

    hyperparams = {}
    units = [400, 25, 10]
    network = OriginNeuralNetwork(units=units, activation='sigmoid', L2_penalty=0.0 )
    hyperparams = network.get_hyperparams()
    # test cost function
    network.set_params(params)
    y_prob,_ = network.forward(X)
    # test cost function
    cost = compute_cost(params,hyperparams,Y,y_prob)
    print('The cost value on test data is: ',cost)
    # test forward propagation
    accuracy = network.score(X,Y)
    print ('The accuracy on hand-written digit dataset is: {}%'.format( accuracy * 100 ))

    # check backward propagation
    check = GradientCheck(network)
    check.checks()

    #fitBGD(network,X,Y)
    #fitMBGD(network,X,Y)
    #fitMBGD2(network,X,Y)
    #fitMomentum(network,X,Y)
    fitNAG(network,X,Y)
    fitRMSprop(network,X,Y)
