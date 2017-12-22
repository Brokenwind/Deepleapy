import sys
sys.path.append('../')
from neural import *
from datamap import *

def load_params():
    params = {}
    data = np.load('hand_written_digit.npz')
    theta1 = data['theta1']
    theta2 = data['theta2']
    b1 = theta1[:,0]
    b1 = b1.reshape((b1.size),1)
    b2 = theta2[:,0]
    b2 = b2.reshape((b2.size),1)
    W1 = np.delete(theta1,0,axis=1)
    W2 = np.delete(theta2,0,axis=1)
    params['b1'] = b1
    params['W1'] = W1
    params['b2'] = b2
    params['W2'] = W2

    return params

if __name__ == '__main__':
    map = DataMap([1,2,3,4,5,6,7,8,9,10])
    data = np.load('hand_written_digit.npz')
    X = data['X']
    X = X.T
    y = data['y']
    Y = map.class2matrix(y)

    params = load_params()
    hyperparams = {}
    units = [400, 25, 10]
    hyperparams['units'] = units
    hyperparams['lamda'] = 0.1

    cost = compute_cost(params,hyperparams,X,Y)
    print('The cost value on test data is: ',cost)

    Y1 = predict(params,hyperparams,X,Y)
    y1 = map.matrix2index(Y1)
    y0 = map.matrix2index(Y)

    print (y0[0:100])
    print (y1[0:100])

    print ( np.sum(y1 == y0)/5000 )

    check_gradient()
