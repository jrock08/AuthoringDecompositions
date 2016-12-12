import numpy as np
import tensorflow as tf

import aiutils.tftools.images as images
import aiutils.tftools.layers as layers
import aiutils.tftools.batch_normalizer as batch_normalizer

class BaseNetwork(object):
    def __init__():
        pass

    """ Call this after setup, while in the correct scope """
    def accumulate_variables(self):
        scope_name = tf.get_variable_scope().name
        assert scope_name != '', 'You almost certainly wanted to be in a variable scope when you created this network'

        # Only care about trainable variables
        self.vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope_name)

class Network(BaseNetwork):
    def __init__(self, input, phase_train=None):
        self.layers = []
        self.layerdict = {}
        #self.batch_size = int(input.get_shape()[0])
        self.add_('input', input)
        if phase_train is not None:
            self.phase_train = phase_train
        else:
            self.phase_train = tf.placeholder_with_default(True, (), 'PhaseTrain')
        self.setup()
        self.accumulate_variables()

    def setup(self):
        raise NotImplementedError('Must be subclassed.')

    def get_unique_name_(self, prefix):
        id = sum(t.startswith(prefix) for t,_ in self.layers)+1
        return '%s_%d'%(prefix, id)

    def add_(self, name, var):
        self.layers.append((name, var))
        self.layerdict[name] = var

    def get_output(self):
        return self.layers[-1][1]

    def conv(self, transpose=False, **args):
        if not transpose:
            self.conv_(**args)
        else:
            self.conv_transpose_(**args)

    def conv_(self, out_dim=-1, stride=1, filter_size=3, func=tf.nn.relu, padding='SAME', batch_norm=False, name=None, inp_layer=None):
        name = name or self.get_unique_name_('conv')
        if inp_layer is None:
            inp_layer = self.get_output()

        new_layer = layers.conv2d(inp_layer, filter_size, out_dim, name, strides=[1,stride,stride,1], func=None, padding=padding)

        if batch_norm:
            new_layer = layers.batch_norm(new_layer, self.phase_train, name=name+'bn')

        if func is not None:
            new_layer = func(new_layer)

        self.add_(name, new_layer)
        return self

    def conv_transpose_(self, out_dim=-1, stride=1, filter_size=3, func=tf.nn.relu, padding='SAME', batch_norm=False, name=None, inp_layer=None, out_shape=None):

        assert out_dim > 0
        name = name or self.get_unique_name_('conv_transpose')
        if inp_layer is None:
            inp_layer = self.get_output()

        new_layer = layers.conv2d_transpose(inp_layer, filter_size, out_dim, name, strides=[1,stride,stride,1], func=None, padding=padding, out_shape=out_shape)

        if batch_norm:
            new_layer = layers.batch_norm(new_layer, self.phase_train, name=name+'bn')

        if func is not None:
            new_layer = func(new_layer)

        self.add_(name, new_layer)
        return self

