import sys
sys.path.append('../')
sys.path.append('../../')
from neural import *
from check import *

def load_data():
    fpath = '../../../dataset/sign_recognition.npz'
    data = np.load(fpath)
    X = data['X']/255.
    y = data['y']
    Xtest = data['Xtest']/255.
    ytest = data['ytest']
    classes = data['classes']

    return X, y, Xtest, ytest, classes

def check_cnn():
    check = GradientCheck()
    check.check_softmax()

if __name__ == '__main__':
    """
    net = LeNet5()
    net.algorithm_init()
    X, y, Xtest, ytest, classes = load_data()
    map = DataMap(classes)
    Y = map.index2matrix(y)
    Y = Y.T
    res, caches = net.forward(X[0:10])
    grads, cost = net.backward(X[0:10],Y[0:10])

    print("cost = "+str(cost))
    """
    
    check_cnn()
