from network import Network
from sklearn import mixture

import tensorflow as tf
import numpy as np
import math

import aiutils.tftools.layers as layers

class LayerConfig():
    def __init__(self):
        self.layer_type = None
        pass

    def layer_type(self):
        return self.layer_type

    def param_set(self):
        raise NotImplementedError('Must be subclassed')

class ConvLayerConfig(LayerConfig):
    def __init__(self, filter_size=3, out_dim = 32, stride=1, padding='SAME', batch_norm=True, func=tf.nn.relu, transpose=False, name=None, out_shape=None):
        self.layer_type = 'conv'
        self.batch_norm = batch_norm
        self.func = func
        self.filter_size = filter_size
        self.out_dim = out_dim
        self.stride = stride
        self.padding = padding
        self.name = name
        self.transpose = transpose
        self.out_shape = out_shape

    def func_lookup(self):
        if self.func == 'lrelu':
            return lambda x: tf.maximum(.01*x,x)
        elif self.func == 'relu':
            return tf.nn.relu
        else:
            return tf.nn.relu
    def param_set(self):
        val = {'batch_norm': self.batch_norm,
              'func': self.func_lookup() if type(self.func) == type('abc') else self.func,
              'filter_size': self.filter_size,
              'out_dim': self.out_dim,
              'padding': self.padding,
              'stride': self.stride,
              'transpose': self.transpose,
              'name':self.name}
        if self.transpose:
            val['out_shape'] = self.out_shape
        return val

class CodeConfig:
    def __init__(self, code_size=12, int_size=512, imsize=None, verify_imsize=True):
        self.code_size = code_size
        self.int_size = int_size
        self.final_imsize = imsize or -1
        self.verify_imsize = verify_imsize
        self.im_nch = 0

    def get_image_size(self, input_layer):

        self.im_nch = int(input_layer.get_shape()[-1])

        if self.final_imsize == -1:
            self.final_imsize = int(input_layer.get_shape()[1])
        else:
            if self.verify_imsize:
                assert self.final_imsize == int(input_layer.get_shape()[1])

    def get_intermediate_code(self):
        """Convolve to a nx1x1xint_size tensor"""
        assert(self.final_imsize)
        return ConvLayerConfig(filter_size=self.final_imsize,
                out_dim=self.int_size,
                padding='VALID',
                name='intermediate_code',
                func=None)

    def get_mean_code(self):
        """Fully connect an nx1x1x? tensor to a nx1x1xcode_size tensor, used to compute code mean"""
        return ConvLayerConfig(filter_size=1, out_dim=self.code_size, func=None, name='mean')

    def get_code_covariance(self):
        """Fully connect an nx1x1x? tensor to a nx1x1xcode_size^2 tensor, used to compute code covariance"""
        return ConvLayerConfig(filter_size=1, out_dim=self.code_size, func=tf.exp, name='covariance')


    def get_decode_config(self):
        """
        First this convolves (fully connect) to create an nx1x1xint_size batch
        Then it performs a transpose conv to create a im_size x im_size x nch image.

        params:
        int_size is the intermediate code size.
        im_size and im_nch can be set to pretty much anything, but you probably want to match up the encoder and decoder.
        """
        assert(self.final_imsize)
        assert(self.im_nch)

        code_size = self.code_size
        int_size = self.int_size
        final_imsize = self.final_imsize
        im_nch = self.im_nch

        return [ConvLayerConfig(filter_size=1, out_dim=int_size, func=None),
                ConvLayerConfig(filter_size=final_imsize, out_dim=im_nch, padding='VALID', transpose=True, func=None)]


class VAEEncoder(Network):
    def __init__(self, input, layer_configs, phase_train=None, code_config=None):
        self.layer_configs = layer_configs
        self.code_config = code_config

        super(VAEEncoder, self).__init__(input, phase_train=phase_train)

    def get_code_config(self):
        return self.code_config

    def setup(self):
        for config in self.layer_configs:
            self.conv(**config.param_set())

        #if not self.code_config.final_imsize:
        self.code_config.get_image_size(self.get_output())


        self.conv(**self.code_config.get_intermediate_code().param_set())
        int_code = self.get_output()

        # Mean
        self.conv(inp_layer=int_code, **self.code_config.get_mean_code().param_set())

        # Covariance
        self.conv(inp_layer=int_code, **self.code_config.get_code_covariance().param_set())

    def full_code(self):
        return self.layerdict['intermediate_code']

    def mean(self):
        return self.layerdict['mean']

    def covariance(self):
        return self.layerdict['covariance']

class VAEDecoder(Network):
    def __init__(self, input, layer_configs, phase_train=None, code_config=None):
        self.layer_configs = layer_configs
        self.code_config = code_config

        super(VAEDecoder, self).__init__(input, phase_train=phase_train)

    def setup(self):
        for config in self.code_config.get_decode_config():
            self.conv(**config.param_set())

        for config in self.layer_configs:
            self.conv(**config.param_set())

    def image(self):
        return self.get_output()

class GMM(Network):
    def __init__(self, input, scipy_gmm, phase_train=None):
        self.scipy_gmm = scipy_gmm
        # This actually checks that we are using the newest version of sklearn as well as the right object type.
        assert(type(scipy_gmm) == mixture.GaussianMixture)

        super(GMM, self).__init__(input, phase_train=phase_train)

    def get_log_prob(self, data):
        # TODO(jrock): rewrite full and tied for better tensorflow usage
        means = self.scipy_gmm.means_
        chol = self.scipy_gmm.precisions_cholesky_
        if self.scipy_gmm.covariance_type == 'diag':
            precisions = chol ** 2
            return (tf.constant(np.sum((means ** 2 * precisions), 1), dtype=tf.float32) -
                    2. * tf.matmul(data, tf.constant((means * precisions).T, dtype=tf.float32)) +
                    tf.matmul(data**2, tf.constant(precisions.T, dtype=tf.float32)))
        elif self.scipy_gmm.covariance_type == 'full':
            log_prob = []
            for k, (mu, prec) in enumerate(zip(means, chol)):
                y = tf.matmul(data, tf.constant(prec, dtype=tf.float32)) - tf.constant(np.dot(mu, prec), dtype=tf.float32)
                log_prob.append(tf.reduce_sum(y**2, 1, keep_dims=True))
            return tf.concat(1, log_prob)
        elif self.scipy_gmm.covariance_type == 'tied':
            log_prob = []
            tf_chol = tf.constant(chol, dtype=tf.float32)
            for k, (mu) in enumerate(means):
                y = tf.matmul(data, tf_chol) - tf.constant(np.dot(mu, chol), dtype=tf.float32)
                log_prob.append(tf.reduce_sum(y**2, 1, keep_dims=True))
            return tf.concat(1, log_prob)
        elif self.scipy_gmm.covariance_type == 'spherical':
            precisions = chol ** 2
            return (tf.constant(np.sum(means ** 2, 1) * precisions, dtype=tf.float32) -
                    2. * tf.matmul(data, tf.constant((means.T * precisions), dtype=tf.float32)) +
                    tf.matmul(tf.reduce_sum(data**2,1, keep_dims=True), tf.constant(np.expand_dims(precisions,0), dtype=tf.float32)))

    def get_log_det(self, data):
        return mixture.gaussian_mixture._compute_log_det_cholesky(self.scipy_gmm.precisions_cholesky_,
            self.scipy_gmm.covariance_type, int(data.get_shape()[1]))
        #if self.scipy_gmm.covariance_type == 'diag':
        #    return np.sum(np.log(self.scipy_gmm.precisions_cholesky_), axis=1)

    def setup(self):
        data = self.get_output()
        shape = data.get_shape().as_list()
        assert(len(shape) == 2)
        n_features = float(shape[1])

        log_det_chol = self.get_log_det(data)
        log_prob = self.get_log_prob(data)

        log_weights = tf.constant(np.expand_dims(np.log(self.scipy_gmm.weights_),0), dtype=tf.float32)

        lpr = -.5 * (tf.constant(n_features * np.log(2 * np.pi),dtype=tf.float32) + log_prob) + log_det_chol + log_weights

        lpr_max = tf.reduce_max(lpr,1,keep_dims=True)
        log_prob = tf.log(tf.reduce_sum(tf.exp(lpr-lpr_max),1)) + tf.squeeze(lpr_max)

        self.add_('lpr', lpr)
        self.add_('log_prob', log_prob)

    def lpr(self):
        return self.layerdict['lpr']

    def log_prob(self):
        return self.layerdict['log_prob']


