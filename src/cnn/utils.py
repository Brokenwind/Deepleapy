import sys
sys.path.append('../')
sys.path.append('../../')
import re
import os
import json
import numpy as np
from activation import *

def zero_pad(X,pad):
    """
    Pad with zeros all images of the dataset X. The padding is applied to
    the height and width of an image
    Parameters
    ----------
    X : python numpy array of shape (m, n_H, n_W, n_C)
        representing a batch of m images
    pad : integer
        amount of padding around each image on vertical and horizontal dimensions

    Returns:
    -------
        padded image of shape (m, n_H + 2*pad, n_W + 2*pad, n_C)
    """
    if not isinstance(pad,tuple) and not isinstance(pad,list):
        raise ValueError('pad should be a tuple or list')
    if X.ndim == 2:
        X = np.pad(X, ((pad[0], pad[0]),(pad[1], pad[1])), 'constant',constant_values=(0,0))
    if X.ndim == 3:
        X = np.pad(X, ((pad[0], pad[0]),(pad[1], pad[1]),(0,0)), 'constant',constant_values=(0,0))
    if X.ndim == 4:
        X = np.pad(X, ((0,0),(pad[0], pad[0]),(pad[1], pad[1]),(0,0)), 'constant',constant_values=(0,0))

    return X

def fc_forward(A_in, W, b, layer):
    """
    Full connected layer forward propagation
    """
    # activation function
    activation = layer['activation']
    Z = np.dot(A_in, W.T) + b
    if activation == 'relu':
        A_out = relu(Z)
    elif activation == 'sigmoid':
        A_out = sigmoid(Z)
    elif activation == 'softmax':
        A_out = softmax(Z)
    else:
        raise ValueError('No such activation: %s' % (activation))

    layer['shape'] = A_out.shape
    cache = (Z, A_in, W, b, layer)

    return A_out, cache

def fc_backward(dZ, cache):
    """
    dZ : the gradient of output value
    """
    (Z, A_in, W, b, layer) = cache
    dW = np.dot(dZ.T, A_in)
    db = np.sum(dZ, axis=0, keepdims = True)
    dA_in = np.dot(dZ, W)

    return dA_in, dW, db


def conv_forward(A_in,W,b,layer):
    """
    Implements the forward propagation for a convolution function
    Parameters
    ----------
    A_in : numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
        output activations of the previous layer
    W : numpy array of shape (f, f, n_C_prev, n_C)
        Weights
    b : numpy array of shape (1, 1, n_C)
        Biases
    hparameters -- python dictionary containing "stride" and "pad"
    Returns:
    -------
    Z : conv output, numpy array of shape (m, n_H, n_W, n_C)
    cache -- cache of values needed for the conv_backward() function
    """
    pad=layer['conv_pad']
    stride=layer['conv_stride']
    filter_size = layer['conv_filter_size']
    m = A_in.shape[0]
    outh = 1 + int( ( A_in.shape[1] - filter_size[0] + 2*pad[0] ) / stride[1] )
    outw = 1 + int( ( A_in.shape[2] - filter_size[0] + 2*pad[1] ) / stride[1] )
    outl = W.shape[3]
    # pad A_in with zeros
    A_in_pad = zero_pad(A_in, pad)
    # the number of output layers
    Z = np.zeros((m,outh,outw,outl))
    for i in range(m):
        sample = A_in_pad[i]
        for h in range(outh):
            for w in range(outw):
                for l in range(outl):
                    starth = h * stride[0]
                    endh = starth + filter_size[0]
                    startw = w * stride[1]
                    endw = startw + filter_size[1]
                    Z[i,h,w,l] = np.sum( sample[starth:endh,startw:endw] * W[:,:,:,l] + b[:,:,:,l])

    A_out = Z
    layer['shape'] = A_out.shape
    cache = (Z, A_in, W, b, layer)

    return A_out, cache


def conv_backward(dZ, cache):
    """
    Implement the backward propagation for a convolution function

    Parameters
    ----------
    dZ : gradient of the cost with respect to the output of the conv layer (Z), numpy array of shape (m, outh, outw, outc)
    cache -- cache of values needed for the conv_backward(), output of conv_forward()

    Returns:
    ----------
    dA_in : gradient of the cost with respect to the input of the conv layer (A_in),
               numpy array of shape (m, inh, inw, inc)
    dW : gradient of the cost with respect to the weights of the conv layer (W)
          numpy array of shape (filter_size, filter_size, inc, outc)
    db : gradient of the cost with respect to the biases of the conv layer (b)
          numpy array of shape (1, 1, 1, outc)
    """

    # Retrieve information from "cache"
    (Z, A_in, W, b, layer) = cache

    # Retrieve dimensions from A_in's shape
    (m, inh, inw, inc) = A_in.shape

    # Retrieve dimensions from W's shape
    #(filter_size, filter_size, inc, outc) = W.shape

    # Retrieve information from "layer"
    stride = layer['conv_stride']
    pad = layer['conv_pad']
    filter_size = layer['conv_filter_size']
    # Retrieve dimensions from dZ's shape
    (m, outh, outw, outc) = dZ.shape

    # Initialize dA_in, dW, db with the correct shapes
    dA_in = np.zeros((m, inh, inw, inc))
    dW = np.zeros((filter_size[0], filter_size[1], inc, outc))
    db = np.zeros((1, 1, 1, outc))

    # Pad A_in and dA_in
    A_in_pad = zero_pad(A_in, pad)
    dA_in_pad = zero_pad(dA_in, pad)

    # loop over the training examples
    for i in range(m):
        # loop on the vertical axis of the output volume
        for h in range(outh):
            # loop on the horizontal axis of the output volume
            for w in range(outw):
                # loop over the channels of the output volume
                for c in range (outc):
                    # Find the corners of the current "slice"
                    starth = h * stride[0]
                    endh = starth + filter_size[0]
                    startw = w * stride[1]
                    endw = startw + filter_size[1]
                    # Update gradients with the selected window and the filter's parameters
                    dA_in_pad[i, starth:endh, startw:endw] += W[:,:,:,c] * dZ[i, h, w, c]
                    a_slice = A_in_pad[i, starth:endh, startw:endw]
                    dW[:,:,:,c] += a_slice * dZ[i, h, w, c]
                    db[:,:,:,c] += dZ[i, h, w, c]
        # remove padding section
        dA_in[i, :, :, :] = dA_in_pad[i, pad[0]:inh+pad[0], pad[1]:inw+pad[1]]

    # Making sure your output shape is correct
    assert(dA_in.shape == (m, inh, inw, inc))

    return dA_in, dW, db

def distribute_value(dz, shape):
    """
    Distributes the input value in the matrix of dimension shape

    Parameters:
    ----------
    dz : input scalar
    shape : the shape (outh, outw) of the output matrix for which we want to distribute the value of dz

    Returns:
    ----------
    a : Array of size (outh, outw) for which we distributed the value of dz
    """

    # Retrieve dimensions from shape
    (outh, outw) = shape

    # Compute the value to distribute on the matrix
    average = dz / (outh * outw)

    # Create a matrix where every entry is the "average" value
    a = np.ones(shape) * average

    return a


def pool_forward(A_in, W, b, layer):
    """
    Implements the forward pass of the pooling layer

    Parameters
    ----------
    A_in : numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
        Input data, m samples
    layer : python dictionary
        must containing the following parameters:
        pool_filter_size : int, the size of pool filter
        pool_stride : int, the stride of pool
        pool_mode : string ("max" or "average"), the pooling mode you would like to use

    Returns:
    ----------
    A : a numpy array of shape (m, n_H, n_W, n_C)
        output of the pool layer
    cache : cache used in the backward pass of the pooling layer, contains the input and layer
    """

    # Retrieve dimensions from the input shape
    (m, inh, inw, inc) = A_in.shape

    # Retrieve hyperparameters from "layer"
    filter_size = layer["pool_filter_size"]
    stride = layer["pool_stride"]
    mode = layer["pool_mode"]

    # Define the dimensions of the output
    outh = int(1 + (inh - filter_size[0]) / stride[0])
    outw = int(1 + (inw - filter_size[1]) / stride[1])
    outc = inc

    # Initialize output matrix A
    Z = np.zeros((m, outh, outw, outc))
    # loop over the training examples
    for i in range(m):
        # the current sample to be processed
        sample = A_in[i]
        # loop on the vertical axis of the output volume
        for h in range(outh):
            # loop on the horizontal axis of the output volume
            for w in range(outw):
                # loop over the channels of the output volume
                for c in range (outc):
                    # Find the corners of the current "slice" (≈4 lines)
                    starth = h * stride[0]
                    endh = starth + filter_size[0]
                    startw = w * stride[1]
                    endw = startw + filter_size[1]
                    if mode == "max":
                        Z[i, h, w, c] = np.max(sample[starth:endh, startw:endw, c])
                    elif mode == "average":
                        Z[i, h, w, c] = np.mean(sample[starth:endh, startw:endw, c])

    # Store the input and layer in "cache" for pool_backward()
    A_out = Z
    layer['shape'] = A_out.shape
    cache = (Z, A_in, np.array([]), np.array([]), layer)
    return A_out, cache


def pool_backward(dA_out, cache):
    """
    Implements the backward pass of the pooling layer

    Parameters:
    dA_out : gradient of cost with respect to the output of the pooling layer, same shape as A_out
    cache : cache output from the forward pass of the pooling layer, contains the layer's input and layer
    mode : the pooling mode you would like to use, defined as a string ("max" or "average")

    Returns:
    dA_in : gradient of cost with respect to the input of the pooling layer, same shape as A_in
    """

    (Z, A_in, W, b, layer) = cache
    stride = layer['pool_stride']
    mode = layer['pool_mode']
    filter_size = layer['pool_filter_size']

    # Retrieve dimensions from A_in's shape and dA_out's shape
    m, inh, inw, inc = A_in.shape
    m, outh, outw, outc = dA_out.shape

    # Initialize dA_in with zeros
    dA_in = np.zeros_like(A_in)

    for i in range(m):
        # select training example from A_in
        a_prev = A_in[i]
        for h in range(outh):
            for w in range(outw):
                for c in range(outc):
                    # Find the corners of the current "slice"
                    starth = h * stride[0]
                    endh = starth + filter_size[0]
                    startw = w * stride[1]
                    endw = startw + filter_size[1]
                    # Compute the backward propagation in both modes.
                    if mode == "max":
                        # Use the corners and "c" to define the current slice from a_prev
                        a_prev_slice = a_prev[starth:endh, startw:endw, c]
                        # Create the mask from a_prev_slice
                        mask = a_prev_slice == (np.max(a_prev_slice))
                        # Set dA_in to be dA_in + (the mask multiplied by the correct entry of dA_out)
                        dA_in[i, starth: endh, startw: endw, c] += mask * dA_out[i, h, w, c]
                    elif mode == "average":
                        # Get the value a from dA_out
                        da = dA_out[i, starth, startw, c]
                        # Distribute it to get the correct slice of dA_in. i.e. Add the distributed value of da.
                        dA_in[i, starth: endh, startw: endw, c] += distribute_value(da, filter_size)

    # Making sure your output shape is correct
    assert(dA_in.shape == A_in.shape)

    return dA_in, np.array([]), np.array([])

def load_config_layers(config_file_name='layers.json'):
    root = os.getcwd().split('src')[0]
    file_name = root+'/src/cnn/'+config_file_name
    if not os.path.exists(file_name):
        print(os.getcwd())
        raise IOError('The configuration file layers.json is not existed!!')
    layers = json.load(open(file_name))
    check_layers(layers)

    return layers

def check_layers(layers):
    num = len(layers)
    keys = sorted(layers.keys())
    p=re.compile(r'(\w+)(\d+)')
    res = np.array( re.findall( p, str(keys) ) )
    names = res[:,0]
    digits = res[:,1].astype('int')

    if np.unique(names).size != 1:
        raise ValueError('The spell of "layer" is not correct!')
    if np.sum(digits == np.arange(0,num)) != num:
        raise ValueError('Your layer number is not continuous!')
