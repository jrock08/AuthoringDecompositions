import tensorflow as tf
import numpy as np

def atrous_image_to_patch(image, patch_size, rate):
    d = int(image.get_shape()[-1])
    filter_val = tf.constant(np.eye(d*patch_size**2).reshape([patch_size, patch_size, d, d*patch_size**2]), dtype=tf.float32)
    patch_filter = tf.get_variable('patch_filter', initializer=filter_val)
    out_patches = tf.nn.atrous_conv2d(image, patch_filter, rate, 'VALID')
    return tf.reshape(out_patches, [-1, patch_size, patch_size, d])


def image_to_patch(image, patch_size, stride, mask=None):
    d = int(image.get_shape()[-1])
    filter_val = tf.constant(np.eye(d*patch_size**2).reshape([patch_size, patch_size, d, d*patch_size**2]), dtype=tf.float32)
    patch_filter = tf.get_variable('patch_filter', initializer=filter_val)
    out_patches = tf.nn.conv2d(image, patch_filter, [1,stride,stride,1], 'VALID')
    if mask is not None:
        out_mask_patch = tf.nn.conv2d(mask, patch_filter[:,:,0:1,0:patch_size**2], [1,stride,stride,1], 'VALID')
        mask_sel = tf.greater(tf.reduce_mean(out_mask_patch, [3]), .1)
        print 'mask_sel_shape'
        print mask_sel.get_shape()
        sel_patchs = tf.boolean_mask(out_patches, mask_sel)
        return tf.reshape(sel_patchs, [-1, patch_size, patch_size, d])
    else:
        return tf.reshape(out_patches, [-1, patch_size, patch_size, d])

def match_to_dict_conv(image_as_patches, dictionary, include_counts=False):
    print 'match_to_dict_conv'
    [n,w,h,c] = dictionary.get_shape().as_list()
    #dict_as_filt = tf.transpose(tf.reshape(dictionary, [-1, w*h*c,1,1]))
    dict_as_filt = tf.transpose(tf.reshape(dictionary, [-1, w*h*c]))
    print dict_as_filt.get_shape()

    [n,w,h,c] = image_as_patches.get_shape().as_list()
    #image_flattened = tf.reshape(image_as_patches, [-1,1,1,w*h*c])
    image_flattened = tf.reshape(image_as_patches, [-1,w*h*c])
    print image_flattened.get_shape()

    #pair_dist = -2 * tf.reshape(tf.nn.conv2d(image_flattened, dict_as_filt, [1,1,1,1], 'SAME'), [n, -1])
    pair_dist = -2 * tf.matmul(image_flattened, dict_as_filt)
    print pair_dist.get_shape()

    single_dist = tf.reduce_sum(tf.square(dictionary),[1,2,3])
    distance = single_dist + pair_dist
    print distance.get_shape()

    min_loc = tf.argmin(distance,1)
    print min_loc.get_shape()

    if include_counts:
        y, _, count = tf.unique_with_counts(min_loc)
        return tf.gather(dictionary, min_loc), [y, count]
    else:
        return tf.gather(dictionary, min_loc)

def match_to_dict(image_as_patches, dictionary):
    patch_size = len(image_as_patches.get_shape())+1

    distance = tf.reduce_sum(tf.square(tf.expand_dims(image_as_patches,1) - tf.expand_dims(dictionary,0)), range(2,patch_size))
    min_loc = tf.argmin(distance, 1)
    return tf.gather(dictionary, min_loc)

