import re
import numpy as np

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
    if X.ndim == 2:
        X = np.pad(X, ((pad, pad),(pad, pad)), 'constant',constant_values=(0,0))
    if X.ndim == 3:
        X = np.pad(X, ((pad, pad),(pad, pad),(0,0)), 'constant',constant_values=(0,0))
    if X.ndim == 4:
        X = np.pad(X, ((0,0),(pad, pad),(pad, pad),(0,0)), 'constant',constant_values=(0,0))

    return X

def fc_middle_forward(A_in, W, b, hyperparams):
    """
    Full connected layer forward propagation
    """
    # activation function
    activation = hyperparams['activation']
    Z = np.dot(W, A_in) + b
    if activation == 'relu':
        A_out = relu(Z)
    elif activation == 'sigmoid':
        A_out = sigmoid(Z)
    else:
        raise ValueError('No such activation: %s' % (activation))

    cache = (Z, A_in, W, b, hyperparams)

    return A_out, cache

def fc_out_forward(A_in, W, b, hyperparams):
    out_activation = hyperparams['out_activation']
    Z = np.dot(W, A_in) + b
    if out_activation == 'softmax':
        A_out = softmax(Z)
    elif out_activation == 'sigmoid':
        A_out = sigmoid(Z)
    else:
        raise ValueError('No such out activation: %s' % (activation))

    cache = (Z, A_in, W, b, hyperparams)

    return A_out, cache

def fc_backward(dZ, cache):
    """
    dZ : the gradient of output value
    """
    (Z, A_in, W, b, hyperparams) = cache
    dW = np.dot(dZ, A_in.T)
    db = np.sum(dZ, axis=1, keepdims = True)
    dA_in = np.dot(W.T, dZ)

    return dA_in, dW, db


def conv_forward(A_in,W,b,hyperparams):
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
    pad=hyperparams['conv_pad']
    stride=hyperparams['conv_stride']
    filter_size = W.shape[0]
    m = A_in.shape[0]
    outh = 1 + int( ( A_in.shape[1] - filter_size + 2*pad ) / stride )
    outw = 1 + int( ( A_in.shape[2] - filter_size + 2*pad ) / stride )
    outl = W.shape[3]
    # pad A_in with zeros
    A_in = zero_pad(A_in, pad)
    # the number of output layers
    Z = np.zeros((m,outh,outw,outl))
    for i in range(m):
        sample = A_in[i]
        for h in range(outh):
            for w in range(outw):
                for l in range(outl):
                    starth = h * stride
                    endh = starth + filter_size
                    startw = w * stride
                    endw = startw + filter_size
                    Z[i,h,w,l] = np.sum( sample[starth:endh,startw:endw] * W[:,:,:,l] + b[:,:,:,l])

    cache = (Z, A_in, W, b, hyperparams)
    A_out = Z

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
    (Z, A_in, W, b, hyperparams) = cache

    # Retrieve dimensions from A_in's shape
    (m, inh, inw, inc) = A_in.shape

    # Retrieve dimensions from W's shape
    (filter_size, filter_size, inc, outc) = W.shape

    # Retrieve information from "hyperparams"
    stride = hyperparams['conv_stride']
    pad = hyperparams['conv_pad']

    # Retrieve dimensions from dZ's shape
    (m, outh, outw, outc) = dZ.shape

    # Initialize dA_in, dW, db with the correct shapes
    dA_in = np.zeros((m, inh, inw, inc))
    dW = np.zeros((filter_size, filter_size, inc, outc))
    db = np.zeros((1, 1, 1, outc))

    # Pad A_in and dA_in
    A_in_pad = zero_pad(A_in, pad)
    dA_in_pad = zero_pad(dA_in, pad)

    # loop over the training examples
    for i in range(m):
        # select ith training example from A_in_pad and dA_in_pad
        a_prev_pad = A_in_pad[i]
        # loop on the vertical axis of the output volume
        for h in range(outh):
            # loop on the horizontal axis of the output volume
            for w in range(outw):
                # loop over the channels of the output volume
                for c in range (outc):
                    # Find the corners of the current "slice"
                    starth = h * stride
                    endh = starth + filter_size
                    startw = w * stride
                    endw = startw + filter_size
                    # Update gradients with the selected window and the filter's parameters
                    dA_in_pad[i, starth:endh, startw:endw] += W[:,:,:,c] * dZ[i, h, w, c]
                    a_slice = a_prev_pad[starth:endh, startw:endw]
                    dW[:,:,:,c] += a_slice * dZ[i, h, w, c]
                    db[:,:,:,c] += dZ[i, h, w, c]
        # remove padding section
        dA_in[i, :, :, :] = dA_in_pad[i, pad:-pad, pad:-pad]

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


def pool_forward(A_in, hyperparams):
    """
    Implements the forward pass of the pooling layer

    Parameters
    ----------
    A_in : numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
        Input data, m samples
    hyperparams : python dictionary
        must containing the following parameters:
        pool_filter_size : int, the size of pool filter
        pool_stride : int, the stride of pool
        pool_mode : string ("max" or "average"), the pooling mode you would like to use

    Returns:
    ----------
    A : a numpy array of shape (m, n_H, n_W, n_C)
        output of the pool layer
    cache : cache used in the backward pass of the pooling layer, contains the input and hyperparams
    """

    # Retrieve dimensions from the input shape
    (m, inh, inw, inc) = A_in.shape

    # Retrieve hyperparameters from "hyperparams"
    filter_size = hyperparams["pool_filter_size"]
    stride = hyperparams["pool_stride"]
    mode = hyperparams["pool_mode"]

    # Define the dimensions of the output
    outh = int(1 + (inh - filter_size) / stride)
    outw = int(1 + (inw - filter_size) / stride)
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
                    starth = h * stride
                    endh = starth + filter_size
                    startw = w * stride
                    endw = startw + filter_size
                    if mode == "max":
                        Z[i, h, w, c] = np.max(sample[starth:endh, startw:endw, c])
                    elif mode == "average":
                        Z[i, h, w, c] = np.mean(sample[starth:endh, startw:endw, c])

    # Store the input and hyperparams in "cache" for pool_backward()
    cache = (Z, A_in, hyperparams)
    A_out = Z
    return A_out, cache


def pool_backward(dA_out, cache):
    """
    Implements the backward pass of the pooling layer

    Parameters:
    dA_out : gradient of cost with respect to the output of the pooling layer, same shape as A_out
    cache : cache output from the forward pass of the pooling layer, contains the layer's input and hyperparams
    mode : the pooling mode you would like to use, defined as a string ("max" or "average")

    Returns:
    dA_in : gradient of cost with respect to the input of the pooling layer, same shape as A_in
    """

    (Z, A_in, hyperparams) = cache
    stride = hyperparams['pool_stride']
    mode = hyperparams['pool_mode']
    filter_size = hyperparams['pool_filter_size']

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
                    starth = h * stride
                    endh = starth + filter_size
                    startw = w * stride
                    endw = startw + filter_size
                    # Compute the backward propagation in both modes.
                    if mode == "max":
                        # Use the corners and "c" to define the current slice from a_prev
                        a_prev_slice = a_prev[starth:endh, startw:endw, c]
                        # Create the mask from a_prev_slice
                        mask = a_prev_slice == (np.max(a_prev_slice))
                        # Set dA_in to be dA_in + (the mask multiplied by the correct entry of dA_out)
                        dA_in[i, starth: endh, startw: endw, c] += mask * dA_out[i, starth, startw, c]
                    elif mode == "average":
                        # Get the value a from dA_out
                        da = dA_out[i, starth, startw, c]
                        # Define the shape of the filter as fxf
                        shape = (filter_size, filter_size)
                        # Distribute it to get the correct slice of dA_in. i.e. Add the distributed value of da.
                        dA_in[i, starth: endh, startw: endw, c] += distribute_value(da, shape)

    # Making sure your output shape is correct
    assert(dA_in.shape == A_in.shape)

    return dA_in

def load_config_layers():
    file_name = 'layers.json'
    if not os.path.exists(file_name):
        raise IOError('The configuration file layers.json is not existed!!')
    layers = json.load(open('layers.json'))
    if not check_layers(layers):
        raise ValueError('The values in configuration file layers.json are not correct!')
    return layers

def check_layers(layers):
    num = len(layers)
    keys = sorted(layers.keys())
    p=re.compile(r'(\w+)(\d+)')
    res = np.array( re.findall( p, str(keys) ) )
    names = res[:,0]
    digits = res[:,1]
    digits = digits.astype('int')
    if np.unique(names).size != 1:
        print('The spell of "layer" is not correct!')
        return False
    if np.sum(digits == np.arange(1,num+1)) != num:
        print('Your layer number is not continuous!')
        return False

    return True
