import tensorflow as tf
import numpy as np

def total_variance(patches, mask=None):
    [n,w,h,c] = patches.get_shape().as_list()

    dx = tf.square(patches[:,1:,:h-1,:] - patches[:,:w-1,:h-1,:])
    dy = tf.square(patches[:,:w-1,1:,:] - patches[:,:w-1,:h-1,:])

    if mask is None:
        return tf.reduce_mean(tf.reduce_mean(tf.sqrt(dx+dy+1e-5),[1,2,3]))
    else:
        m = mask[:,1:,1:,:]
        return tf.reduce_mean(tf.reduce_mean(tf.reduce_sum(tf.sqrt(dx+dy+1e-5)*m,[1,2])/tf.maximum(1.0, tf.reduce_sum(m,[1,2])),[1]))


def image_prior(patches, mask=None):
    [n,w,h,c] = patches.get_shape().as_list()

    dx = tf.abs(patches[:,1:,:h-1,:] - patches[:,:w-1,:h-1,:])
    dy = tf.abs(patches[:,:w-1,1:,:] - patches[:,:w-1,:h-1,:])

    if mask is None:
        return tf.reduce_mean(tf.reduce_mean(dx + dy, [1,2,3]))
    else:
        m = mask[:,1:,1:,:]
        return tf.reduce_mean(tf.reduce_mean(tf.reduce_sum((dx+dy)*m,[1,2])/tf.maximum(1.0, tf.reduce_sum(m, [1,2])),[1]))

def similarity_fro(albedo_as_patches, shading_as_patches):
    [n,w,h,c] = albedo_as_patches.get_shape().as_list()
    [n_,w_,h_,c_] = shading_as_patches.get_shape().as_list()

    assert(w==w_ and  h==h_ and c==c_ and n==n_)

    image = tf.concat(3, [albedo_as_patches, shading_as_patches])
    image = tf.reshape(image, [-1, w*h, 2*c])
    mean_image, _ = tf.nn.moments(image, [1], keep_dims=True)
    im_centered = image-mean_image

    covariance = tf.reduce_mean(tf.expand_dims(im_centered,-1) * tf.expand_dims(im_centered,-2), 1)
    cov = covariance[:,:c,c:]
    return tf.reduce_mean(tf.reduce_sum(tf.square(cov), [1,2]))
