from neural import *
from check import *

def load_params():
    params = {}
    data = np.load('hand_written_digit.npz')
    X = data['X']
    y = data['y']
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

def load_data():
    # load and rearrange data
    x = np.loadtxt('x.txt')
    #x = np.hstack((np.ones((np.size(x,0),1)),x))
    y = np.loadtxt('y.txt')

    return x.T, expandY(y,10)

if __name__ == '__main__':
    X,y = load_data()
    params = load_params()
    hyperparams = {}
    units = [400, 25, 10]
    hyperparams['units'] = units
    hyperparams['lamda'] = 0.1
    cost = compute_cost(params,hyperparams,X,y)
    print('The cost value on test data is: ',cost)
    pred = predict(params,hyperparams,X,y)
    yo = np.argmax(y,axis=0)+1
    print (pred)
    print (yo)
    print ( np.sum(pred == yo)/5000 )
