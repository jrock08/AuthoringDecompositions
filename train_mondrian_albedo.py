import sys
sys.path.append('pyAIUtils')

import scipy.misc
import scipy.cluster
import numpy as np
import tensorflow as tf

import network.graphs as graphs
import network.models as models

import constants
import utils

flags = tf.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('run_id', 'mondrian', '')
flags.DEFINE_string('root_id', '_n', '')

lrelu = 'lrelu'
# Encoding which more carefully matches the face encoding
encode_config = [[models.ConvLayerConfig(filter_size=5, out_dim = 64, stride=2, padding='SAME', batch_norm=True, func=lrelu),
                  models.ConvLayerConfig(filter_size=5, out_dim = 64, stride=2, padding='SAME', batch_norm=True, func=lrelu),
                  models.ConvLayerConfig(filter_size=3, out_dim = 64, stride=1, padding='SAME', batch_norm=True, func=lrelu)],
                 [models.ConvLayerConfig(filter_size=5, out_dim = 64, stride=2, padding='SAME', batch_norm=True, func=lrelu),
                  models.ConvLayerConfig(filter_size=5, out_dim = 64, stride=2, padding='SAME', batch_norm=True, func=lrelu),
                  models.ConvLayerConfig(filter_size=3, out_dim = 64, stride=1, padding='SAME', batch_norm=True, func=lrelu)],
                 [models.ConvLayerConfig(filter_size=5, out_dim = 64, stride=2, padding='SAME', batch_norm=True, func=lrelu),
                  models.ConvLayerConfig(filter_size=5, out_dim = 64, stride=2, padding='SAME', batch_norm=True, func=lrelu),
                  models.ConvLayerConfig(filter_size=3, out_dim = 64, stride=1, padding='SAME', batch_norm=True, func=lrelu)],
                 [models.ConvLayerConfig(filter_size=5, out_dim = 64, stride=2, padding='SAME', batch_norm=True, func=lrelu),
                  models.ConvLayerConfig(filter_size=5, out_dim = 64, stride=2, padding='SAME', batch_norm=True, func=lrelu),
                  models.ConvLayerConfig(filter_size=3, out_dim = 64, stride=1, padding='SAME', batch_norm=True, func=lrelu)]]


decode_config = [[models.ConvLayerConfig(filter_size=3, out_dim = 64, stride=1, padding='SAME', batch_norm=False, func=lrelu, transpose=True),
                  models.ConvLayerConfig(filter_size=5, out_dim = 64, stride=2, padding='SAME', batch_norm=False, func=lrelu, transpose=True),
                  models.ConvLayerConfig(filter_size=5, out_dim = 64, stride=2, padding='SAME', batch_norm=False, func=lrelu, transpose=True),
                  models.ConvLayerConfig(filter_size=5, out_dim = 3, stride=1, padding='SAME', batch_norm=False, func=None)],
                 [models.ConvLayerConfig(filter_size=3, out_dim = 64, stride=1, padding='SAME', batch_norm=False, func=lrelu, transpose=True),
                  models.ConvLayerConfig(filter_size=5, out_dim = 64, stride=2, padding='SAME', batch_norm=False, func=lrelu, transpose=True),
                  models.ConvLayerConfig(filter_size=5, out_dim = 64, stride=2, padding='SAME', batch_norm=False, func=lrelu, transpose=True),
                  models.ConvLayerConfig(filter_size=5, out_dim = 3, stride=1, padding='SAME', batch_norm=False, func=None)],
                 [models.ConvLayerConfig(filter_size=3, out_dim = 64, stride=1, padding='SAME', batch_norm=False, func=lrelu, transpose=True),
                  models.ConvLayerConfig(filter_size=5, out_dim = 64, stride=2, padding='SAME', batch_norm=False, func=lrelu, transpose=True),
                  models.ConvLayerConfig(filter_size=5, out_dim = 64, stride=2, padding='SAME', batch_norm=False, func=lrelu, transpose=True),
                  models.ConvLayerConfig(filter_size=5, out_dim = 3, stride=1, padding='SAME', batch_norm=False, func=None)],
                 [models.ConvLayerConfig(filter_size=3, out_dim = 64, stride=1, padding='SAME', batch_norm=False, func=lrelu, transpose=True),
                  models.ConvLayerConfig(filter_size=5, out_dim = 64, stride=2, padding='SAME', batch_norm=False, func=lrelu, transpose=True),
                  models.ConvLayerConfig(filter_size=5, out_dim = 64, stride=2, padding='SAME', batch_norm=False, func=lrelu, transpose=True),
                  models.ConvLayerConfig(filter_size=5, out_dim = 3, stride=1, padding='SAME', batch_norm=False, func=None)]]


def print_images(ims, pred, iter_num, out_dir, out_type):
    utils.print_images(ims, '/im_%s_%d'%(out_type, iter_num), out_dir)
    utils.print_images(np.exp(np.minimum(-pred,0.0)),'/pred_%s_%d'%(out_type, iter_num), out_dir)

const = constants.Constants(FLAGS.root_id, FLAGS.run_id)
const.num_train_epochs = 500
const.vae_code_size = [4] * 4
const.vae_int_code_size = [64] * 4
const.vae_int_image_size = [4] * 4
const.patch_size = 64*2
const.vae_regression_weight = 1000
const.vae_total_variation_weight = .1
const.vae_kl_div_weight = 10.0
const.vae_kl_div_sig_mult = .02
const.make_dirs()
utils.save_params(const, const.model_dir(), 'const')

print_every = 500

image_shape = const.default_image_data_shape()

d_train = const.get_mondrian_albedo_data()
d_test = const.get_augmented_data()[1]

vae_train_params = graphs.VAETrainParams()
vae_train_params.intermediate_code_size = const.vae_int_code_size
vae_train_params.code_size = const.vae_code_size
vae_train_params.image_size = const.vae_int_image_size
vae_train_params.kl_div_weight = const.vae_kl_div_weight
vae_train_params.kl_div_sig_mult = const.vae_kl_div_sig_mult
vae_train_params.regression_weight = const.vae_regression_weight
vae_train_params.full_regression_weight = None
vae_train_params.total_variaton_weight = const.vae_total_variation_weight
vae_train_params.summary_output = const.summary_dir()
vae_train_params.normalize_images=True
vae_train_params.init_lr = .001
vae_train_params.lr_decay = .9
vae_train_params.lr_decay_steps = 500
vae_train_params.fine_tune_decoder = False
vae_train_params.pyr_size = 3
vae_train_params.encode_config = encode_config
vae_train_params.decode_config = decode_config

utils.save_params(vae_train_params, const.model_dir(), 'params_albedo')

vae_train = graphs.VAELaplacianTrainNoClass(image_shape, params=vae_train_params)

for i, ims in enumerate(d_train):
    print vae_train.minimize(ims['image'], ims['mask'], 1.0)

    if i%print_every == print_every-1:
        # Train recon
        recon = vae_train.generate_predictions(ims['image'], ims['mask'], 0.0)
        print_images(ims['image'], recon, i, const.vae_out_dir(), 'albedo')

        # Test Recon
        ims = d_test.next()
        recon = vae_train.generate_predictions(ims['albedo'], ims['mask'], 0.0)
        print_images(ims['albedo'], recon, i+1, const.vae_out_dir(), 'albedo')
        vae_train.save(const.model_dir() + '/%s_vae'%('albedo_gen'))

vae_train.save(const.model_dir() + '/%s_vae'%('albedo_gen'))
