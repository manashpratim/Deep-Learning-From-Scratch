# %%
# Part II Numpy implementations
import numpy as np
import hw2_utils


def convolution(x, w, b, stride, pad):
    """
    4.1 Understanding Convolution
    Forward Pass for a convolution layer.
    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width HH.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
    - 'pad': The number of pixels that will be used to zero-pad the input.

    Output:
    - out: Output of the forward pass
    """

    hw2_utils.exercise(
        andrew_username="mbarman", # <<< set your andrew username here
        seed=42
    )

    out = None

    ###########################################################################
    # Your code starts here
    ###########################################################################
    # TODO 4.1 Understanding Convolution

    x_pad = []
    for i in range(x.shape[0]):
        l = []
        for j in range(x.shape[1]):
            l.append(np.pad(x[i,j,:,:], ((pad, pad), (pad, pad)), 'constant', constant_values=0))
        x_pad.append(l)

    x_pad = np.array(x_pad)


    h = int((x.shape[2] - w.shape[2] + (2*pad))/stride) + 1
    t = int((x.shape[3] - w.shape[3] + (2*pad))/stride) + 1
    out = np.zeros((x.shape[0],w.shape[0],h,t))

   
    for m in range(out.shape[0]):
        for f in range(w.shape[0]):
            for c in range(w.shape[1]):
                p = 0
                for i in range(out.shape[2]):
                    q = 0
                    for j in range(out.shape[3]):
                        out[m,f,i,j] = out[m,f,i,j] + np.sum(x_pad[m,c,p:p+w.shape[2],q:q+w.shape[3]]*w[f,c,:,:])
                        q = q + stride
                    p = p + stride
            out[m,f,:,:] = out[m,f,:,:] + b[f]



    ###########################################################################
    # END OF YOUR CODE
    ###########################################################################

    return out


def grayscale_filter(w):
    """ your grayscale filter

    Modify the second filter of the input filters as a grayscale filter

    Input:
    - w: Conv filter of shape [2, 3, 3, 3]

    Output:
    - w: The modified filter

    """

    hw2_utils.exercise(
        andrew_username="mbarman", # <<< set your andrew username here
        seed=42
    )

    ###########################################################################
    # Your code starts here
    ###########################################################################

    w[1, 0, :, 1] = [0, 0.299, 0]
    w[1, 1, :, 1] = [0, 0.587, 0]
    w[1, 2, :, 1] = [0, 0.114, 0]

    ###########################################################################
    # END OF YOUR CODE
    ###########################################################################

    return w


def relu(x):
    """
    4.2 ReLU Implementation
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    """

    hw2_utils.exercise(
        andrew_username="mbarman", # <<< set your andrew username here
        seed=42
    )

    out = None

    ###########################################################################
    # Your code starts here
    ###########################################################################
    # TODO: 4.2 ReLU Implementation

    out = np.maximum(x,0)

    ###########################################################################
    # END OF YOUR CODE
    ###########################################################################

    return out


def max_pool(x, pool_param):
    """
    4.3 MaxPooling Implementation
    A naive implementation of the forward pass for a max-pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    Returns a tuple of:
    - out: Output data
    """

    hw2_utils.exercise(
        andrew_username="mbarman", # <<< set your andrew username here
        seed=42
    )

    out = None

    ###########################################################################
    # Your code starts here
    ###########################################################################
    # TODO: 4.3 MaxPooling Implementation

    h = int((x.shape[2] - pool_param['pool_height'])/pool_param ['stride']) + 1
    w = int((x.shape[3] - pool_param['pool_width'])/pool_param ['stride']) + 1
    out = np.zeros((x.shape[0],x.shape[1],h,w))

    for c in range(x.shape[0]):
        for d in range(x.shape[1]):
            p =0
            for i in range(out.shape[2]):
                q = 0
                for j in range(out.shape[3]):
                    out[c,d,i,j] = np.max(x[c,d,p:p+pool_param['pool_height'],q:q+pool_param['pool_width']])
                    q = q + pool_param['stride']
                p = p + pool_param['stride']

    ###########################################################################
    # END OF YOUR CODE
    ###########################################################################

    return out


def dropout(x, mode, p):
    """
    4.4  Dropout Implementation
    Performs the forward pass for (inverted) dropout.
    Inputs:
    - x: Input data, of any shape
    - p: Dropout parameter. We drop each neuron output with probability p.
    - mode: 'test' or 'train'. If the mode is train, then perform dropout;
    if the mode is test, then just return the input.
    Outputs:
    - out: Array of the same shape as x.
    """

    hw2_utils.exercise(
        andrew_username="mbarman", # <<< set your andrew username here
        seed=42
    )

    out = None

    if mode == 'train':
        #######################################################################
        # Your code starts here
        #######################################################################
        # TODO: 4.3 MaxPooling Implementation

        keep_probability = 1 - p
        mask = np.random.uniform(0, 1.0, x.shape) < keep_probability
        if keep_probability > 0.0:
           scale = (1/keep_probability)
        else:
           scale = 0.0
        out =  mask*x*scale

        #######################################################################
        # END OF YOUR CODE
        #######################################################################

    elif mode == 'test':
        #######################################################################
        # Your code starts here
        #######################################################################
        # TODO: 4.3 Test mode of dropout

        out = x

        #######################################################################
        # END OF YOUR CODE
        #######################################################################

    return out
