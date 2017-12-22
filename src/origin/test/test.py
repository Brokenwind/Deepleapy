import sys
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

if __name__ == '__main__':
    fpath = '../../../dataset/hand_written_digit.npz'
    map = DataMap([1,2,3,4,5,6,7,8,9,10])
    data = np.load(fpath)
    params = load_params(fpath)

    X = data['X']
    y = data['y']
    Y = map.class2matrix(y)


    hyperparams = {}
    units = [400, 25, 10]
    hyperparams['units'] = units
    hyperparams['activition'] = 'sigmoid'
    hyperparams['lamda'] = 0.1

    cost = compute_cost(params,hyperparams,X,Y)
    print('The cost value on test data is: ',cost)

    Y1 = predict(params,hyperparams,X,Y)
    y1 = map.matrix2index(Y1)
    y0 = map.matrix2index(Y)

    accuracy = np.sum(y1 == y0)/5000.0
    print ('The accuracy on hand-written digit dataset is: {}%'.format( accuracy * 100 ))

    check_gradient()
