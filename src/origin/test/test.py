import sys
import time
sys.path.append('../')
sys.path.append('../../')
from neural import *
from datamap import *

def load_params(fpath):
    params = {}
    data = np.load(fpath)
    params['b1'] = data['b1']
    params['W1'] = data['W1']
    params['b2'] = data['b2']
    params['W2'] = data['W2']

    return params

def fitparams(X,Y):
    hyperparams = {}
    units = [400, 25, 10]
    hyperparams['units'] = units
    hyperparams['activition'] = 'relu'
    hyperparams['classifier'] = 'sigmoid'
    hyperparams['lamda'] = 1.0
    hyperparams['iters'] = 500
    hyperparams['alpha'] = 0.2

    tic = time.process_time()

    params = gradient_descent(hyperparams,X,Y)

    toc = time.process_time()
    print ("----- Computation time = " + str(1000*(toc - tic)) + "ms")

    # test forward propagation
    Y1 = predict(params,hyperparams,X,Y)
    y1 = map.matrix2index(Y1)
    y0 = map.matrix2index(Y)
    accuracy = np.sum(y1 == y0)/5000.0
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
    hyperparams['units'] = units
    hyperparams['activition'] = 'sigmoid'
    hyperparams['classifier'] = 'sigmoid'
    hyperparams['lamda'] = 0.0
    # test cost function
    cost = compute_cost(params,hyperparams,X,Y)
    print('The cost value on test data is: ',cost)
    # test forward propagation
    Y1 = predict(params,hyperparams,X,Y)
    y1 = map.matrix2index(Y1)
    y0 = map.matrix2index(Y)
    accuracy = np.sum(y1 == y0)/5000.0
    print ('The accuracy on hand-written digit dataset is: {}%'.format( accuracy * 100 ))
    # check backward propagation
    check_gradient()

    fitparams(X,Y)
