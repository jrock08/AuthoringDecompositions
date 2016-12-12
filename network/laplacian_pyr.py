import tensorflow as tf
import numpy as np

def get_filter(inp_ch = 3):
    g = np.array([1.0,4.0,6.0,4.0,1.0])/16
    f = np.outer(g,g)
    size = g.shape[0]
    filter = np.zeros([size, size, inp_ch, inp_ch])
    for i in range(inp_ch):
        filter[:,:,i,i] = f

    return tf.constant(filter, dtype=tf.float32)


def create_pyr(image, levels=3):
    """
    The 0th image is the high frequency stuff, the levels-th image is the gaussian small image.
    """
    tf_filt = get_filter()

    I = list()
    I.append(image)
    for i in range(1,levels+1):
        I.append(tf.nn.conv2d(I[i-1], tf_filt, [1,2,2,1], 'SAME'))

    G = list()
    for i in range(0,levels):
        G.append(I[i] - tf.nn.conv2d_transpose(I[i+1], tf_filt, I[i].get_shape(), [1,2,2,1]))
    G.append(I[-1])

    return G

def invert_pyr(pyr, full_output = False):
    tf_filt = get_filter()

    pyr_r = pyr[-1::-1]
    I = list()
    I.append(pyr_r[0])
    for i in range(1,len(pyr)):
        I.append(pyr_r[i] + tf.nn.conv2d_transpose(I[i-1], tf_filt, pyr_r[i].get_shape(), [1,2,2,1]))

    if full_output:
        return I[::-1]
    else:
        return I[-1]

def create_gauss_pyr(image, levels=3):
    nch = int(image.get_shape()[-1])
    tf_filt = get_filter(nch)

    I = list()
    I.append(image)
    for i in range(1,levels+1):
        I.append(tf.nn.conv2d(I[i-1], tf_filt, [1,2,2,1], 'SAME'))

    return I
