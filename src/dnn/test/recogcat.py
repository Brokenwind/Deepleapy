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
    y = y.reshape((1,y.size))
    ytest = ytest.reshape((1,ytest.size))
    return X, y, Xtest, ytest, classes

if __name__ == '__main__':
    X,y,Xtest,ytest,classes = load_data()
    Xnorm,mu,std = normalize(X,axis=1)
    Xtest = (Xtest - mu)/std
    map = DataMap(classes)

    hyperparams = {}
    units = [12288,20,7,5,1]
    hyperparams['units'] = units
    hyperparams['lossfunc'] = 'log_loss'
    # the activition of layer 1 - L-1. L means how many layers of the network except the first input layer
    hyperparams['activation'] = 'relu'
    hyperparams['solver'] = 'MBGD'
    # the last layer activiation
    hyperparams['out_activation'] = 'sigmoid'
    hyperparams['L2_penalty'] = 3.
    hyperparams['max_iters'] = 300
    hyperparams['learning_rate_init'] = 0.001

    network = OriginNeuralNetwork(units=units)
    network.set_hyperparams(hyperparams)

    params = network.fit(Xnorm,y)

    # test on trainning dataset
    accuracy = network.score(Xnorm,y)
    print ('The accuracy on trainning dataset is: {}%'.format( accuracy * 100 ))

    # test on test dataset
    accuracy = network.score(Xtest,ytest)
    print ('The accuracy on teset dataset is: {}%'.format( accuracy * 100 ))
