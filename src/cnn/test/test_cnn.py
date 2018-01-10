import sys
sys.path.append('../')
sys.path.append('../../')
from neural import *

def load_data():
    fpath = '../../../dataset/sign_recognition.npz'
    data = np.load(fpath)
    X = data['X']
    y = data['y']
    Xtest = data['Xtest']
    ytest = data['ytest']
    classes = data['classes']

    return X, y, Xtest, ytest, classes

if __name__ == '__main__':
    net = LeNet5()
    net.algorithm_init()
    X, y, Xtest, ytest, classes = load_data()
    map = DataMap(classes)
    Y = map.index2matrix(y)
    Y = Y.T
    res, caches = net.forward(X[0:2])
    net.backward(X[0:2],Y[0:2])
