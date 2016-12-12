import sys
sys.path.append('pyAIUtils')

import os

import scipy.cluster
import scipy.ndimage
import sklearn.mixture
import numpy as np
import tensorflow as tf
import random

import network.graphs as graphs

import constants
import utils

flags = tf.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('run_id', 'decomp_albedo_shading', '')
flags.DEFINE_string('root_id', '_n', '')

flags.DEFINE_bool('encode_train', True, '')
flags.DEFINE_bool('print_intermediate', False, '')

def read_and_encode_images(vae_encode, image_generator, name):
    code = []
    for i,ims in enumerate(image_generator):
        means = vae_encode.generate_prediction_cat(ims[name])
        Q = []
        for j,m in enumerate(means):
            if i == 0:
                code.append([np.reshape(m, [-1, m.shape[-1]])])
            else:
                code[j].append(np.reshape(m, [-1, m.shape[-1]]))

    out_code = []
    for c in code:
        out_code.append(np.concatenate(c, 0))

    return out_code

def cropims(ims, crops):
    return ims[:,:crops[0], :crops[1], :]

def print_preds(pred, mask, image_idx, iter_num, out_dir, crops):
    suffix = '%d_%d'%(image_idx, iter_num) if iter_num >= 0 else '%d'%(image_idx)
    utils.print_images(cropims(np.exp(np.minimum(0, -pred[0])) * mask, crops),'/shading_%s'%(suffix), out_dir)
    utils.print_images(cropims(np.exp(np.minimum(0, -pred[1])) * mask, crops),'/albedo_%s'%(suffix), out_dir)
    utils.print_images(cropims((np.exp(-pred[2]) - .5) * mask, crops),'/residual_%s'%(suffix), out_dir)

def print_images(ims, pred, mask, image_idx, iter_num, out_dir, crops):
    suffix = '%d_%d'%(image_idx, iter_num) if iter_num >= 0 else '%d'%(image_idx)
    utils.print_images(cropims(ims * mask,crops), '/im_%s'%(suffix), out_dir)
    print_preds(pred, mask, image_idx, iter_num, out_dir, crops)

def print_gt(albedo, shading, mask, image_idx, out_dir, crops):
    utils.print_images(cropims(albedo*mask, crops), '/albedo_gt_%d'%(image_idx), out_dir)
    utils.print_images(cropims(shading*mask, crops), '/shading_gt_%d'%(image_idx), out_dir)
    utils.print_images(cropims(np.tile(mask,[1,1,1,3]), crops), '/mask_%d'%(image_idx), out_dir)

const_ = constants.Constants(FLAGS.root_id, FLAGS.run_id)

const = utils.load_params(const_.model_dir(), 'const')
const.run_id = FLAGS.run_id
const.root_id = FLAGS.root_id
const.assign_data_dir()

const.make_pred_dirs()

pred_params = graphs.PredictionParams()
pred_params.smart_init = True
pred_params.learning_rate = .15
pred_params.decay_steps = 500

pred_params.nmeans = 1
pred_params.cov_type = 'spherical'
pred_params.reconstruction_error_weight = 100.0
pred_params.logprob_weight = [.0001, .0001]
pred_params.corr_patch_size = [5]
pred_params.corr_patch_rate = [1]
pred_params.patch_agreement_weight = [10000.0, 10000.0, 10000.0, 10000.0]
pred_params.relative_recon_err = False
pred_params.code_initializer = 'random_normal'

utils.save_params(pred_params, const.pred_dir(), 'pred_params')
pred_params.code_initializer = tf.random_normal_initializer(0.0, 1.0)

image_shape = const.default_image_data_shape()
vae_shading_train_params = utils.load_params(const.model_dir(), 'params_shading')
vae_albedo_train_params = utils.load_params(const.model_dir(), 'params_albedo')

image_shape = [image_shape[0], 2*image_shape[1], 2*image_shape[2], image_shape[3]]

if FLAGS.encode_train:
    print "encoding"

    vae_encode = graphs.VAEEncodeOnlyLaplacian(image_shape, params=vae_shading_train_params, name='')
    vae_encode.restore(const.model_dir() + '/shading_gen_vae')

    d_train = const.get_rendered_shading_data(num_epochs=5, patch_size=image_shape[1])
    code_sh = read_and_encode_images(vae_encode, d_train, 'image')

    vae_encode = graphs.VAEEncodeOnlyLaplacian(image_shape, params=vae_albedo_train_params, name='')
    vae_encode.restore(const.model_dir() + '/albedo_gen_vae')

    d_train = const.get_mondrian_albedo_data(num_epochs = 10, patch_size = image_shape[1])
    code_albedo = read_and_encode_images(vae_encode, d_train, 'image')

    np.savez(const.encode_dir() + '/albedo_one', *code_albedo)
    np.savez(const.encode_dir() + '/sh_one', *code_sh)
else:
    with open(const.encode_dir() + '/albedo_one.npz') as f:
        f_alb = np.load(f)
        code_albedo = []
        for i in range(len(f_alb.files)):
            code_albedo.append(f_alb['arr_%d'%(i)])
    with open(const.encode_dir() + '/sh_one.npz') as f:
        f_sh = np.load(f)
        code_sh = []
        for i in range(len(f_alb.files)):
            code_sh.append(f_sh['arr_%d'%(i)])


gmm_material = []
gmm_sh = []
gmm_albedo = []

for i in range(len(code_albedo)):
    # GMM STUFF
    code_sh_ = np.random.permutation(code_sh[i])
    code_albedo_ = np.random.permutation(code_albedo[i])

    sh_gmm = code_sh_[:100000]
    albedo_gmm = code_albedo_[:100000]

    gmm_sh.append(sklearn.mixture.GaussianMixture(n_components=pred_params.nmeans, covariance_type=pred_params.cov_type, max_iter=100, n_init=10))
    gmm_sh[-1].fit(sh_gmm)

    gmm_albedo.append(sklearn.mixture.GaussianMixture(n_components=pred_params.nmeans, covariance_type=pred_params.cov_type, max_iter=100, n_init=10))
    gmm_albedo[-1].fit(albedo_gmm)

# START DECOMP CODE
d_test = const.get_orig_datamgr(dtype='test')

for image_idx in range(10):
    ims_ = d_test.ims(image_idx)
    w = ims_.image.shape[0]
    crop_w = w
    while w%64 != 0:
        w+=1
    h = ims_.image.shape[1]
    crop_h = h
    while h%64 != 0:
        h+=1


    ims_ = ims_.crop_image([0,0], [w,h])

    ims = {}
    ims['albedo'] = np.expand_dims(ims_.albedo,0).astype(np.float32)
    ims['shading'] = np.expand_dims(ims_.shading,0).astype(np.float32)
    ims['image'] = np.expand_dims(ims_.image,0).astype(np.float32)
    ims['mask'] = np.expand_dims(np.expand_dims(ims_.mask,0),-1).astype(np.float32)

    if pred_params.smart_init:
        # Initialize smartly
        im = ims['image'] * (ims['mask'] + (1-ims['mask']) * .01)
        shading_init = scipy.ndimage.gaussian_filter(im, sigma=5)
        albedo_init = im / np.maximum(shading_init, .01)

        vae_encode = graphs.VAEEncodeOnlyLaplacian(ims['image'].shape, params=vae_albedo_train_params, name='')
        vae_encode.restore(const.model_dir() + '/albedo_gen_vae')
        albedo_code = vae_encode.generate_predictions(albedo_init)

        vae_encode = graphs.VAEEncodeOnlyLaplacian(ims['image'].shape, params=vae_shading_train_params, name='')
        vae_encode.restore(const.model_dir() + '/shading_gen_vae')
        shading_code = vae_encode.generate_predictions(shading_init)

        im_pred = graphs.VAEPrediction(ims['image'], ims['mask'],
            [gmm_sh, gmm_albedo],
            [vae_shading_train_params, vae_albedo_train_params], pred_params=pred_params,
            initial_codes = [shading_code, albedo_code])
    else:
        im_pred = graphs.VAEPrediction(ims['image'], ims['mask'],
            [gmm_sh, gmm_albedo],
            [vae_shading_train_params, vae_albedo_train_params], pred_params=pred_params)

    im_pred.restore([const.model_dir() + '/shading_gen_vae',
        const.model_dir() + '/albedo_gen_vae'])

    for o in range(10):
        if FLAGS.print_intermediate:
            im_out = im_pred.get_image()
            print_preds(im_out, ims['mask'], image_idx, o, const.pred_dir(), [crop_w, crop_h])

        for i in range(100):
            min_out = im_pred.minimize()
        print im_pred.minimize_print()

    im_out = im_pred.get_image()
    print_images(ims['image'], im_out, ims['mask'], image_idx, -1, const.pred_dir(), [crop_w, crop_h])

    albedo_gt = ims['albedo']
    shading_gt = ims['shading']

    print_gt(albedo_gt, shading_gt, ims['mask'], image_idx, const.pred_dir(), [crop_w, crop_h])
