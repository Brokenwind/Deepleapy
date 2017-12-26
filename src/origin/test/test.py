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
    hyperparams['tol'] = 0.01
    hyperparams['max_iters'] = 500
    hyperparams['learning_rate_init'] = 0.2

    network.set_hyperparams(hyperparams)
    tic = time.process_time()
    params = network.fit(X,Y)
    toc = time.process_time()
    print ("----- Computation time = " + str(1000*(toc - tic)) + "ms")

    # test forward propagation
    accuracy = network.score(X,Y)
    print ('The accuracy on hand-written digit dataset is: {}%'.format( accuracy * 100 ))


def fitMBGD(network,X,Y):
    hyperparams = {}
    units = [400, 25, 10]
    hyperparams['solver'] = 'MBGD'
    hyperparams['units'] = units
    hyperparams['lossfunc'] = 'log_loss'
    hyperparams['activation'] = 'relu'
    hyperparams['out_activation'] = 'sigmoid'
    hyperparams['L2_penalty'] = 1.0
    hyperparams['max_iters'] = 200
    hyperparams['tol'] = 0.01
    hyperparams['learning_rate_init'] = 0.15
    hyperparams['batch_size'] = 512

    network.set_hyperparams(hyperparams)
    tic = time.process_time()
    params = network.fit(X,Y)
    toc = time.process_time()
    print ("----- Computation time = " + str(1000*(toc - tic)) + "ms")

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
    print (y_prob.shape)
    # test cost function
    cost = compute_cost(params,hyperparams,Y,y_prob)
    print('The cost value on test data is: ',cost)
    # test forward propagation
    accuracy = network.score(X,Y)
    print ('The accuracy on hand-written digit dataset is: {}%'.format( accuracy * 100 ))

    # check backward propagation
    check = GradientCheck(network)
    check.check_gradient()

    fitBGD(network,X,Y)
    fitMBGD(network,X,Y)
