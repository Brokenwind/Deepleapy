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
    y = y.reshape((1,y.size))
    ytest = ytest.reshape((1,ytest.size))
    print(X.shape)
    return X, y, Xtest, ytest, classes

if __name__ == '__main__':
    net = LeNet5()
    net.algorithm_init()
    X, y, Xtest, ytest, classes = load_data()
    res, caches = net.forward(X[0:2])
    print(res)
    print(y[0,0:2])
