import sys
sys.path.append('pyAIUtils')

import os

import scipy.cluster
import sklearn.mixture
import numpy as np
import tensorflow as tf
import random

import network.graphs as graphs

import constants
import utils

flags = tf.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('run_id', 'decomp_ASM', '')
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

def resize_ims(ims, multiplier):
    ims_ = []
    for i in range(ims.shape[0]):
        ims_.append(scipy.ndimage.zoom(ims[i], [multiplier] * 2 + [1], order=1))
    return np.asarray(ims_)

def cropims(ims, crops):
    return ims[:,:crops[0], :crops[1], :]

def print_preds(pred, mask, image_idx, iter_num, out_dir, crops, multiplier):
    suffix = '%d_%d'%(image_idx, iter_num) if iter_num >= 0 else '%d'%(image_idx)
    utils.print_images(resize_ims(cropims(np.exp(np.minimum(0, -pred[0])) * mask, crops), multiplier), '/shading_%s'%(suffix), out_dir)
    utils.print_images(resize_ims(cropims(np.exp(np.minimum(0, -pred[1])) * mask, crops), multiplier), '/material_%s'%(suffix), out_dir)
    utils.print_images(resize_ims(cropims(np.exp(np.minimum(0, -pred[2])) * mask, crops), multiplier),'/albedo_%s'%(suffix), out_dir)
    utils.print_images(resize_ims(cropims((np.exp(-pred[3]) - .5) * mask, crops), multiplier), '/residual_%s'%(suffix), out_dir)

def print_images(ims, pred, mask, image_idx, iter_num, out_dir, crops, multiplier):
    suffix = '%d_%d'%(image_idx, iter_num) if iter_num >= 0 else '%d'%(image_idx)
    utils.print_images(resize_ims(cropims(ims * mask, crops), multiplier), '/im_%s'%(suffix), out_dir)
    print_preds(pred, mask, image_idx, iter_num, out_dir, crops, multiplier)

def print_gt(albedo, shading, mask, image_idx, out_dir):
    utils.print_images(albedo*mask, '/albedo_gt_%d'%(image_idx), out_dir)
    utils.print_images(shading*mask, '/shading_gt_%d'%(image_idx), out_dir)
    utils.print_images(np.tile(mask,[1,1,1,3]), '/mask_%d'%(image_idx), out_dir)

const_ = constants.Constants(FLAGS.root_id, FLAGS.run_id)

const = utils.load_params(const_.model_dir(), 'const')
const.run_id = FLAGS.run_id
const.root_id = FLAGS.root_id
const.assign_data_dir()

const.make_pred_dirs()

pred_params = graphs.PredictionParams()
pred_params.learning_rate = .1

pred_params.code_initializer = tf.random_normal_initializer(0.0, 1.0)
pred_params.nmeans = 1
pred_params.cov_type = 'spherical'
pred_params.reconstruction_error_weight = 10000.0
pred_params.logprob_weight = [.0001] * 3
pred_params.corr_patch_size = [5]
pred_params.corr_patch_rate = [1]
pred_params.patch_agreement_weight = [10000.0, 10000.0, 10000.0, 10000.0]
pred_params.relative_recon_err = False
pred_params.code_initializer = 'random_normal'
pred_params.smart_init = True

utils.save_params(pred_params, const.pred_dir(), 'pred_params')
pred_params.code_initializer = tf.random_normal_initializer(0.0, 1.0)

image_shape = const.default_image_data_shape()
vae_shading_train_params = utils.load_params(const.model_dir(), 'params_shading')
vae_material_train_params = utils.load_params(const.model_dir(), 'params_material')
vae_albedo_train_params = utils.load_params(const.model_dir(), 'params_albedo')

image_shape = [image_shape[0], 2*image_shape[1], 2*image_shape[2], image_shape[3]]

if FLAGS.encode_train:
    print "encoding"

    vae_encode = graphs.VAEEncodeOnlyLaplacian(image_shape, params=vae_shading_train_params, name='')
    vae_encode.restore(const.model_dir() + '/shading_gen_vae')

    d_train = const.get_rendered_shading_data(num_epochs=10, patch_size=image_shape[1])
    code_sh = read_and_encode_images(vae_encode, d_train, 'image')

    vae_encode = graphs.VAEEncodeOnlyLaplacian([45] + image_shape[1:], params=vae_material_train_params, name='')
    vae_encode.restore(const.model_dir() + '/material_gen_vae')

    d_train = const.get_material_data(num_epochs = 20, batch_size=45, patch_size = image_shape[1])
    code_material = read_and_encode_images(vae_encode, d_train, 'image')

    vae_encode = graphs.VAEEncodeOnlyLaplacian(image_shape, params=vae_albedo_train_params, name='')
    vae_encode.restore(const.model_dir() + '/albedo_gen_vae')

    d_train = const.get_mondrian_albedo_data(num_epochs = 10, patch_size = image_shape[1])
    code_albedo = read_and_encode_images(vae_encode, d_train, 'image')

    np.savez(const.encode_dir() + '/albedo_one', *code_albedo)
    np.savez(const.encode_dir() + '/material_one', *code_material)
    np.savez(const.encode_dir() + '/sh_one', *code_sh)
else:
    with open(const.encode_dir() + '/albedo_one.npz') as f:
        f_alb = np.load(f)
        code_albedo = []
        for i in range(len(f_alb.files)):
            code_albedo.append(f_alb['arr_%d'%(i)])
    with open(const.encode_dir() + '/material_one.npz') as f:
        f_alb = np.load(f)
        code_material = []
        for i in range(len(f_alb.files)):
            code_material.append(f_alb['arr_%d'%(i)])
    with open(const.encode_dir() + '/sh_one.npz') as f:
        f_sh = np.load(f)
        code_sh = []
        for i in range(len(f_alb.files)):
            code_sh.append(f_sh['arr_%d'%(i)])


gmm_material = []
gmm_sh = []
gmm_albedo = []

for i in range(len(code_material)):
    # GMM STUFF
    code_sh_ = np.random.permutation(code_sh[i])
    code_material_ = np.random.permutation(code_material[i])
    code_albedo_ = np.random.permutation(code_albedo[i])

    sh_gmm = code_sh_[:100000]
    material_gmm = code_material_[:100000]
    albedo_gmm = code_albedo_[:100000]

    gmm_sh.append(sklearn.mixture.GaussianMixture(n_components=pred_params.nmeans, covariance_type=pred_params.cov_type, max_iter=100, n_init=10))
    gmm_sh[-1].fit(sh_gmm)

    gmm_material.append(sklearn.mixture.GaussianMixture(n_components=pred_params.nmeans, covariance_type=pred_params.cov_type, max_iter=100, n_init=10))
    gmm_material[-1].fit(material_gmm)

    gmm_albedo.append(sklearn.mixture.GaussianMixture(n_components=pred_params.nmeans, covariance_type=pred_params.cov_type, max_iter=100, n_init=10))
    gmm_albedo[-1].fit(albedo_gmm)

# START ACTUAL CODE
d_test = const.get_orig_datamgr(dtype='test')
for image_idx in range(10):
    ims_ = d_test.ims(image_idx)
    im_orig = {}
    im_orig['albedo'] = np.expand_dims(ims_.albedo,0).astype(np.float32)
    im_orig['shading'] = np.expand_dims(ims_.shading,0).astype(np.float32)
    im_orig['image'] = np.expand_dims(ims_.image,0).astype(np.float32)
    im_orig['mask'] = np.expand_dims(np.expand_dims(ims_.mask,0),-1).astype(np.float32)

    # This handles the one case (Sun) that the input image is a little too big for one TitanX.
    # This only happens in the 3-way decomposition.
    w = ims_.albedo.shape[0]
    h = ims_.albedo.shape[1]
    use_resize = False
    if w > 8*64 or h > 8*64:
        use_resize = True
        multiplier = min(8.0*64/w, 8.0*64/h)
        ims_ = ims_.scale_image_size([multiplier, multiplier])

    print ims_.albedo.shape

    w = ims_.albedo.shape[0]
    crop_w = w
    while w%64 != 0:
        w+=1
    h = ims_.albedo.shape[1]
    crop_h = h
    while h%64 != 0:
        h+=1

    print 'imsize %d %d'%(w,h)
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

        vae_encode = graphs.VAEEncodeOnlyLaplacian(ims['albedo'].shape, params=vae_material_train_params, name='')
        vae_encode.restore(const.model_dir() + '/material_gen_vae')
        material_code = vae_encode.generate_predictions(np.zeros_like(ims['albedo']) + .5)
        vae_encode = None

        vae_encode = graphs.VAEEncodeOnlyLaplacian(ims['shading'].shape, params=vae_shading_train_params, name='')
        vae_encode.restore(const.model_dir() + '/shading_gen_vae')
        shading_code = vae_encode.generate_predictions(shading_init)
        vae_encode = None

        vae_encode = graphs.VAEEncodeOnlyLaplacian(ims['albedo'].shape, params=vae_albedo_train_params, name='')
        vae_encode.restore(const.model_dir() + '/albedo_gen_vae')
        albedo_code = vae_encode.generate_predictions(albedo_init)
        # Try to make sure this clears off the gpu
        vae_encode = None

        im_pred = graphs.VAEPrediction(ims['image'], ims['mask'],
            [gmm_sh, gmm_material, gmm_albedo],
            [vae_shading_train_params, vae_material_train_params, vae_albedo_train_params], pred_params=pred_params,
            initial_codes = [shading_code, material_code, albedo_code],
            constant_shift = np.log(2.0))
    else:
        im_pred = graphs.VAEPrediction(ims['image'], ims['mask'],
            [gmm_sh, gmm_material, gmm_albedo],
            [vae_shading_train_params, vae_material_train_params, vae_albedo_train_params], pred_params=pred_params,
            constant_shift = np.log(2.0))

    im_pred.restore([const.model_dir() + '/shading_gen_vae',
        const.model_dir() + '/material_gen_vae',
        const.model_dir() + '/albedo_gen_vae'])

    albedo_gt = im_orig['albedo']
    shading_gt = im_orig['shading']

    print_gt(albedo_gt, shading_gt, im_orig['mask'], image_idx, const.pred_dir())

    for o in range(10):
        if FLAGS.print_intermediate:
            im_out = im_pred.get_image()
            print_preds(im_out, ims['mask'], image_idx, o, const.pred_dir(), [crop_w, crop_h], 1/multiplier if use_resize else 1.0)

        for i in range(50):
            min_out = im_pred.minimize()
        print im_pred.minimize_print()

    im_out = im_pred.get_image()
    print_images(ims['image'], im_out, ims['mask'], image_idx, -1, const.pred_dir(), [crop_w, crop_h], 1/multiplier if use_resize else 1.0)

