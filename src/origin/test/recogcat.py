import sys
sys.path.append('../')
sys.path.append('../../')
from neural import *
from datamap import *
from prodata import *

def load_data():
    fpath = '../../../dataset/cat_recognization.npz'
    data = np.load(fpath)
    X = data['X']
    y = data['y']
    Xtest = data['Xtest']
    ytest = data['ytest']
    classes = data['classes']
    X = X.reshape(X.shape[0],-1).T
    Xtest = Xtest.reshape(Xtest.shape[0],-1).T

    return X, y, Xtest, ytest, classes

if __name__ == '__main__':
    X,y,Xtest,ytest,classes = load_data()
    Xnorm,mu,std = normalize(X)
    Xtest = (Xtest - mu)/std
    map = DataMap(classes)

    hyperparams = {}
    units = [12288,20,7,5,1]
    hyperparams['units'] = units
    hyperparams['activition'] = 'relu'
    hyperparams['lamda'] = 1.
    hyperparams['iters'] = 1000
    hyperparams['alpha'] = 0.3

    params = gradient_descent(hyperparams,Xnorm,y)

    # test on trainning dataset
    py = predict(params,hyperparams,Xnorm,y)
    py = map.matrix2index(py)

    accuracy = np.sum(py == y)/y.size
    print ('The accuracy on trainning dataset is: {}%'.format( accuracy * 100 ))

    # test on test dataset
    pytest = predict(params,hyperparams,Xtest,ytest)
    pytest = map.matrix2index(pytest)
    accuracy = np.sum(pytest == ytest)/ytest.size
    print ('The accuracy on teset dataset is: {}%'.format( accuracy * 100 ))

