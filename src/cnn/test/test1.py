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
    hparameters = {"pad" : 2, "stride": 2}
    Z = conv(A_prev, W, b, pad=2, stride=2)
    print(Z.shape)
    print("Z's mean = ", np.mean(Z))
    print("Z[3,2,1] =", Z[3,2,1])

if __name__ == '__main__':
    test_conv()
