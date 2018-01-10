import sys
sys.path.append('..')
import numpy as np
from utils import *

def test_conv():
    """
    expected results:
    0.0562764082167
    0.0734406101506
    0.0139133071457
    (10, 4, 4, 8)
    Z's mean =  0.202041582131
    Z[3,2,1] = [  3.48211796  -8.37731144  -4.56894114  13.867986    12.1423875
  -6.39349327  -7.06247915   8.01747414]
    """
    np.random.seed(1)
    A_prev = np.random.randn(10,4,4,3)
    W = np.random.randn(2,2,3,8)
    b = np.random.randn(1,1,1,8)
    print(np.mean(A_prev))
    print(np.mean(W))
    print(np.mean(b))
    hyperparams = {"conv_pad" : [2,2], "conv_stride": [2,2], "conv_filter_size": [2,2]}
    Z,cache = conv_forward(A_prev, W, b, hyperparams)
    print(Z.shape)
    print("Z's mean = ", np.mean(Z))
    print("Z[3,2,1] =", Z[3,2,1])

def test_pool():
    """
    expected results:
    max mode:
    [[[[ 1.74481176  1.6924546   2.10025514]]]

    [[[ 1.19891788  1.51981682  2.18557541]]]]

    average mode:
    [[[[-0.09498456  0.11180064 -0.14263511]]]

    [[[-0.09525108  0.28325018  0.33035185]]]]
    """
    np.random.seed(1)
    A_prev = np.random.randn(2, 4, 4, 3)
    hyperparams = {"pool_stride" : [1,1], "pool_filter_size": [4,4]}

    hyperparams['pool_mode'] = 'max'
    A, cache = pool_forward(A_prev, None, None, hyperparams)
    print("max mode:")
    print(A)

    hyperparams['pool_mode'] = 'average'
    A, cache = pool_forward(A_prev, None, None, hyperparams)
    print("average mode: ")
    print(A)

def test_conv_back():
    """
    expected values:
    dA_mean = 9.60899067587
    dW_mean = 10.5817412755
    db_mean = 76.3710691956
    """
    np.random.seed(1)
    A_prev = np.random.randn(10,4,4,3)
    W = np.random.randn(2,2,3,8)
    b = np.random.randn(1,1,1,8)
    hyperparams = {"conv_pad" : [2,2], "conv_stride": [1,1], "conv_filter_size": [2,2]}
    Z, cache = conv_forward(A_prev, W, b, hyperparams)
    dA, dW, db = conv_backward(Z, cache)
    print("dA_mean =", np.mean(dA))
    print("dW_mean =", np.mean(dW))
    print("db_mean =", np.mean(db))

def test_pool_back():
    """
    expected results:

    mode = max
    mean of dA =  0.145713902729
    dA_prev[1,1] = 
    [[ 0.          0.        ]
    [ 5.05844394 -1.68282702]
    [ 0.          0.        ]]

    mode = average
    mean of dA =  0.145713902729
    dA_prev[1,1] = 
    [[ 0.08485462  0.2787552 ]
    [ 1.26461098 -0.25749373]
    [ 1.17975636 -0.53624893]]
    """
    np.random.seed(1)
    A_prev = np.random.randn(5, 5, 3, 2)
    hyperparams = {"pool_stride" : [1,1], "pool_filter_size": [2,2], "pool_mode":"max"}
    A, cache = pool_forward(A_prev, None, None, hyperparams)
    dA = np.random.randn(5, 4, 2, 2)

    hyperparams["pool_mode"] = "max"
    dA_prev,dW,db = pool_backward(dA, cache)
    print(dA_prev.shape)
    print("mode = max")
    print('mean of dA = ', np.mean(dA))
    print('dA_prev[1,1] = ')
    print(dA_prev[1,1])
    print()

    hyperparams["pool_mode"] = "average"
    dA_prev,dW,db = pool_backward(dA, cache)
    print("mode = average")
    print('mean of dA = ', np.mean(dA))
    print('dA_prev[1,1] = ')
    print(dA_prev[1,1])

def test_cost():
    np.random.seed(1)
    X = np.random.randn(4,64,64,3)
    Y = np.random.randn(4,6)
    Z3=[[-0.44670227, -1.57208765, -1.53049231, -2.31013036, -1.29104376, 0.46852064],[-0.17601591, -1.57972014, -1.4737016, -2.61672091, -1.00810647, 0.5747785 ]]
    res, caches = net.forward(X)
    grads, cost = net.backward(X,Y)

if __name__ == '__main__':
    test_conv()
    test_pool()
    test_conv_back()
    test_pool_back()
