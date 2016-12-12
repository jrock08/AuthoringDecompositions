import tensorflow as tf
import numpy as np
import random
import re
import aiutils.tftools.var_collect as var_collect
from aiutils.tftools.placeholder_management import PlaceholderManager
import dict_matcher
import variance_estimate
import laplacian_pyr

from losses import Losses
from summary_writer import summary_writer

import models
import itertools

def rgb_to_YCgCo(images):
    f = np.array([[.25, .5, .25],[-.25,.5,-.25],[.5,0,-.5]]).T
    f = np.expand_dims(f,0)
    f = np.expand_dims(f,0)

    return tf.nn.conv2d(images, f, [1,1,1,1], 'SAME')

def YCgCo_to_rgb(images):
    f = np.array([[1,-1,1],[1,1,0],[1,-1,-1]]).T
    f = np.expand_dims(f,0)
    f = np.expand_dims(f,0)

    return tf.nn.conv2d(images, f, [1,1,1,1], 'SAME')

class VAETrainParams:
    def __init__(self):
        # Code and stuff
        self.code_size = 32
        self.intermediate_code_size = 512
        self.image_size = 4
        self.normalize_images = False
        self.encode_config = None
        self.decode_config = None
        self.pyr_size = 0

        # Losses
        self.kl_div_weight = 1.0
        self.kl_div_sig_mult = -1

        self.regression_weight = 1.0
        self.full_regression_weight = 1.0
        self.res_regression_weight = .1
        self.total_variation_weight = .1

        # Optimizer
        self.init_lr = .1
        self.lr_decay = .5
        self.lr_decay_steps = 250

        self.summary_output = '/tmp/summary/VAE_classifier'

class PredictionParams:
    def __init__(self):
        #self.code_initializer = tf.constant_initializer(0.0)
        self.code_initializer = tf.random_normal_initializer(0.0, 1.0)
        self.corr_patch_size = [5, 7, 7, 7]
        self.corr_patch_rate = [1, 2, 3, 4]

        self.alb_tv_weight = 1.0
        self.sh_tv_weight = None
        self.reconstruction_error_weight = 10.0
        self.logprob_sh_weight = .1
        self.logprob_alb_weight = .1
        self.patch_agreement_weight = [1000.0, 0.0, 0.0]
        self.nmeans = 1
        self.cov_type = 'full'
        self.relative_recon_err = False
        self.low_bias = [6.0, 6.0, 6.0]
        self.parsimony_weight = 1.0
        self.parsimony_bins = 100
        self.parsimony_low_high = [-.5, 5.0]
        self.parsimony_sim_variance_mult = 6.0
        self.parsimony_sim_gaussian_size = 10

        self.num_mask_clusters = 3

        self.decay_steps = None
        self.learning_rate = None

class VAEEncodeOnlyLaplacian:
    def __init__(self, image_shape, params=VAETrainParams(), name=None):
        self.graph = tf.Graph()
        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        self.sess = tf.Session(graph=self.graph, config=sess_config)

        encode_config = params.encode_config
        decode_config = params.decode_config

        n,w,h,c = image_shape
        #assert(w==h)

        with self.graph.as_default():
            self.pl = PlaceholderManager()

            self.pl.add_placeholder('image', tf.float32, image_shape)
            self.pl.add_placeholder('phase_train', tf.bool, [])

            log_im = -tf.log(tf.maximum(self.pl['image'], .01))
            if params.normalize_images:
                log_im = rgb_to_YCgCo(log_im)

            im_pyr = laplacian_pyr.create_pyr(log_im, params.pyr_size)

            self.mean = []
            for i in range(len(im_pyr)):
                with tf.variable_scope('VAE_%s_%d'%(name, i) if name else 'VAE_%d'%(i)):
                    with tf.variable_scope('Encoder'):
                        encoder_model = models.VAEEncoder(im_pyr[i], encode_config[i], phase_train=self.pl['phase_train'], code_config = models.CodeConfig(code_size=params.code_size[i], int_size=params.intermediate_code_size[i], imsize=params.image_size[i], verify_imsize=False))
                        self.mean.append(encoder_model.mean())

            full_size = self.mean[0].get_shape().as_list()
            self.cat_means = [self.mean[0]]
            for i in range(1,len(self.mean)):
                self.cat_means.append(tf.image.resize_images(self.mean[i], full_size[1:3], tf.image.ResizeMethod.NEAREST_NEIGHBOR))

            self.cat_mean = tf.concat(3, self.cat_means)

        tf.initialize_variables(
            var_list=var_collect.collect_all(self.graph)).run(
                session=self.sess)

        self.saver = tf.train.Saver(var_collect.collect_scope('VAE',
                                                              self.graph))

    def restore(self, name):
        self.saver.restore(self.sess, name)

    def generate_predictions(self, image):
        return self.sess.run(self.mean,
            self.pl.get_feed_dict({'phase_train': False,
                                   'image': image}))

    def generate_prediction_cat(self, image):
        return self.sess.run([self.cat_mean],
            self.pl.get_feed_dict({'phase_train': False,
                                   'image': image}))

class VAEPrediction:
    def __init__(self, input_image, mask, gmms, params, pred_params, initial_codes=None, constant_shift = 0.0):
        self.graph = tf.Graph()
        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        self.sess = tf.Session(graph=self.graph, config=sess_config)

        # This graph is used to get sizes of things, it should never be run
        size_graph = tf.Graph()
        with size_graph.as_default():
            log_im = tf.constant(-np.log(np.maximum(input_image, .01)))

            CODE_SHAPE = []
            encoder_models = []
            for p_i, param in enumerate(params):
                im_pyr = laplacian_pyr.create_pyr(log_im, param.pyr_size)
                CODE_SHAPE.append([])
                encoder_models.append([])

                for i in range(len(im_pyr)):
                    with tf.variable_scope('VAE_%d_%d'%(p_i, i)):
                        with tf.variable_scope('Encoder'):
                            code_config = models.CodeConfig(code_size=param.code_size[i], int_size=param.intermediate_code_size[i], imsize=param.image_size[i], verify_imsize=False)
                            encoder_models[p_i].append(models.VAEEncoder(im_pyr[i], param.encode_config[i], code_config = code_config))

                            CODE_SHAPE[p_i].append(encoder_models[-1][-1].mean().get_shape().as_list())

        with self.graph.as_default():
            log_im = tf.constant(-np.log(np.maximum(input_image, .01)))
            mask_pyr = laplacian_pyr.create_gauss_pyr(tf.constant(mask, dtype=tf.float32), 4)

            self.log_probs = []
            self.pyrs = []
            self.gmm = []
            self.pl = PlaceholderManager()
            self.pl.add_placeholder('phase_train', tf.bool, [])

            codes = []

            cov_det = tf.constant(0.0)

            for p_i, param in enumerate(params):
                codes.append([])
                self.pyrs.append([])
                for i in range(param.pyr_size+1):
                    if initial_codes:
                        code = tf.get_variable('code_%d_%d'%(p_i, i), dtype=tf.float32, initializer = initial_codes[p_i][i])
                    else:
                        code = tf.get_variable('code_%d_%d'%(p_i, i), shape=CODE_SHAPE[p_i][i], dtype=tf.float32, initializer = pred_params.code_initializer)
                    codes[p_i].append(code)
                    with tf.variable_scope('VAE_l%d_%d'%(p_i,i)):
                        with tf.variable_scope('Decoder'):
                            decoder_model = models.VAEDecoder(code, param.decode_config[i], phase_train=self.pl['phase_train'], code_config=encoder_models[p_i][i].get_code_config())

                    self.pyrs[p_i].append(decoder_model.image())

            #self.inv_pyrs = []
            #for i in range(len(self.pyrs)):
            #    self.inv_pyrs.append(laplacian_pyr.invert_pyr(self.pyrs[i],True))
            #
            #for i in range(len(self.pyrs)-1):
            #    for j in range(i+1,len(self.pyrs)):
            #        pyr_depth = min(len(self.pyrs[i]), len(self.pyrs[j]))

            #        pyr1 = self.inv_pyrs[i]
            #        pyr2 = self.inv_pyrs[j]

            #        for k in range(pyr_depth):
            #            print pyr1[k].get_shape()
            #            print pyr2[k].get_shape()

            #            for l, [patch_size, rate] in enumerate(zip(pred_params.corr_patch_size, pred_params.corr_patch_rate)):
            #                with tf.variable_scope('patches_%d_%d'%(k,l)) as scope:
            #                    if not(i == 0 and j == 1):
            #                        scope.reuse_variables()

            #                    if rate == 1:
            #                        patch1 = dict_matcher.image_to_patch(pyr1[k], patch_size, 1)
            #                        scope.reuse_variables()
            #                        patch2 = dict_matcher.image_to_patch(pyr2[k], patch_size, 1)
            #                    else:
            #                        patch1 = dict_matcher.atrous_image_to_patch(pyr1[k], patch_size, rate)
            #                        scope.reuse_variables()
            #                        patch2 = dict_matcher.atrous_image_to_patch(pyr2[k], patch_size, rate)

            #                    cov_det += pred_params.patch_agreement_weight[k] * variance_estimate.similarity_fro(patch1, patch2)


            for i in range(len(self.pyrs)-1):
                for j in range(i+1,len(self.pyrs)):
                    if params[i].pyr_size == params[j].pyr_size:
                        pyr1 = self.pyrs[i]
                        pyr2 = self.pyrs[j]
                    elif params[i].pyr_size > params[j].pyr_size:
                        diff = -(params[i].pyr_size - params[j].pyr_size) -1
                        pyr1 = self.pyrs[i][:diff] + [laplacian_pyr.invert_pyr(self.pyrs[i][diff:])]
                        pyr2 = self.pyrs[j]
                    else:
                        diff = params[i].pyr_size - params[j].pyr_size -1
                        pyr1 = self.pyrs[j][:diff] + [laplacian_pyr.invert_pyr(self.pyrs[j][diff:])]
                        pyr2 = self.pyrs[i]

                    for k in range(len(pyr1)):
                        for l, [patch_size, rate] in enumerate(zip(pred_params.corr_patch_size, pred_params.corr_patch_rate)):
                            with tf.variable_scope('patches_%d_%d'%(k,l)) as scope:
                                if not(i == 0 and j == 1):
                                    scope.reuse_variables()

                                if rate == 1:
                                    patch1 = dict_matcher.image_to_patch(pyr1[k], patch_size, 1)
                                    scope.reuse_variables()
                                    patch2 = dict_matcher.image_to_patch(pyr2[k], patch_size, 1)
                                else:
                                    patch1 = dict_matcher.atrous_image_to_patch(pyr1[k], patch_size, rate)
                                    scope.reuse_variables()
                                    patch2 = dict_matcher.atrous_image_to_patch(pyr2[k], patch_size, rate)

                                cov_det += pred_params.patch_agreement_weight[k] * variance_estimate.similarity_fro(patch1, patch2)

            code_rshape = []
            full_code = []
            for c_i, code in enumerate(codes):
                full_size = code[0].get_shape().as_list()
                code_rshape.append([code[0]])
                for i in range(1, len(code)):
                    code_rshape[-1].append(tf.image.resize_images(code[i], full_size[1:3], tf.image.ResizeMethod.NEAREST_NEIGHBOR))
                full_code.append(tf.concat(3, code_rshape[-1]))

                with tf.variable_scope('GMM_%d'%(c_i)):
                    mean_reshape = tf.reshape(full_code[-1], [-1, int(full_code[-1].get_shape()[-1])])
                    self.gmm.append(models.GMM(mean_reshape, gmms[c_i][0], self.pl['phase_train']))

                self.log_probs.append(tf.reduce_mean(self.gmm[-1].log_prob()))

            self.decoded_images = [laplacian_pyr.invert_pyr(pyr) for pyr in self.pyrs]

            self.log_images = [YCgCo_to_rgb(dec) for dec in self.decoded_images]
            self.all_images = self.log_images + [log_im - tf.add_n(self.log_images) + constant_shift]

            with tf.variable_scope('loss'):
                self.losses = Losses(collection=['summaries', 'losses'],
                                 session=self.sess)

                self.valid = tf.add_n([tf.reduce_mean(tf.square(tf.minimum(im, 0)) * mask) for im in self.log_images])
                self.losses.add_loss(self.valid, 'valid', 1.0)

                if pred_params.relative_recon_err:
                    self.reconstruction_error = tf.reduce_mean((tf.square(self.all_images[-1])* mask)/ (tf.square(log_im) + .1))
                else:
                    self.reconstruction_error = tf.reduce_mean(tf.square(self.all_images[-1]) * mask)

                self.losses.add_loss(self.reconstruction_error, 'reconstruction_error', pred_params.reconstruction_error_weight)

                self.log_prob = []
                for i, log_prob in enumerate(self.log_probs):
                    self.log_prob.append(log_prob)

                    self.losses.add_loss(-self.log_prob[-1], 'logprob_%d'%(i), pred_params.logprob_weight[i])

                self.losses.add_loss(cov_det, 'patch_agreement', 1.0)
                self.cov_det = cov_det

                self.loss = self.losses.full_loss()

            with tf.variable_scope('optimizer'):
                global_step = tf.Variable(0, trainable=False)

                lr = tf.train.exponential_decay(pred_params.learning_rate or .25, global_step, pred_params.decay_steps or 100, .9, staircase=True)
                #lr = pred_params.learning_rate or .1
                optimizer = tf.train.AdamOptimizer(lr)

                self.opt = optimizer.minimize(
                    self.loss,
                    var_list = [y for x in codes for y in x],
                    global_step = global_step)

        tf.initialize_variables(
            var_list=var_collect.collect_all(self.graph)).run(
                session=self.sess)


        self.saver = []
        for i in range(len(params)):
            var = var_collect.collect_scope('VAE_l%d_'%(i), self.graph)
            var = {re.sub('_l%d_'%(i), '_', v.op.name): v for v in var}
            self.saver.append(tf.train.Saver(var))

    def restore(self, names):
        for i,name in enumerate(names):
            self.saver[i].restore(self.sess, name)

    def minimize(self):
        return self.sess.run([self.opt], self.pl.get_feed_dict({'phase_train': False}))

    def minimize_print(self):
        return self.sess.run([self.loss] + self.log_prob + [self.cov_det, self.reconstruction_error, self.valid],
                self.pl.get_feed_dict({'phase_train':False}))

    def get_image(self):
        return self.get_decoded_image()

    def get_decoded_image(self):
        return self.sess.run(self.all_images,
                self.pl.get_feed_dict({'phase_train':False}))

class VAELaplacianTrainNoClass:
    def __init__(self, image_shape, params=VAETrainParams()):
        self.params = params
        self.graph = tf.Graph()
        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        sess_config.intra_op_parallelism_threads = 3
        self.sess = tf.Session(graph=self.graph, config=sess_config)
        encode_config = params.encode_config
        decode_config = params.decode_config

        n,w,h,c = image_shape
        assert(w==h)

        with self.graph.as_default():
            self.pl = PlaceholderManager()

            self.pl.add_placeholder('image', tf.float32, image_shape)
            self.pl.add_placeholder('mask', tf.float32, image_shape[:-1] + [1])
            self.pl.add_placeholder('phase_train', tf.bool, [])
            self.pl.add_placeholder('epsilon', tf.float32, [])

            log_im = -tf.log(tf.maximum(self.pl['image'], .01))

            if params.normalize_images:
                log_im = rgb_to_YCgCo(log_im)

            im_pyr = laplacian_pyr.create_pyr(log_im,params.pyr_size)

            self.decoder_albedo_model = []
            self.decoder_joint_model = []
            self.decoder_shading_model = []
            self.out_pyr_images = []
            mean = []
            covariance = []
            for i in range(len(im_pyr)):
                with tf.variable_scope('VAE_%d'%(i)):
                    with tf.variable_scope('Encoder'):
                        encoder_model = models.VAEEncoder(im_pyr[i], encode_config[i], phase_train=self.pl['phase_train'], code_config = models.CodeConfig(code_size=params.code_size[i], int_size=params.intermediate_code_size[i], imsize=params.image_size[i], verify_imsize=False))
                        mean.append(encoder_model.mean())
                        covariance.append(encoder_model.covariance())

                    with tf.variable_scope('ParamTrick'):
                        epsilon = tf.random_normal(covariance[-1].get_shape()) * self.pl['epsilon']
                        new_mean = mean[-1] + epsilon * tf.sqrt(covariance[-1])
                    with tf.variable_scope('Decoder'):
                        decoder_model = models.VAEDecoder(new_mean, decode_config[i], phase_train=self.pl['phase_train'], code_config=encoder_model.get_code_config())

                self.out_pyr_images.append(decoder_model.image())

            self.out_images = laplacian_pyr.invert_pyr(self.out_pyr_images)
            with tf.variable_scope('loss'):
                global_step = tf.Variable(1, trainable=False, name='global_step')

                self.losses = Losses(collection=['summaries', 'losses'],
                                 session=self.sess)
                # losses should be per batch item after this point.
                trace = lambda x: tf.reduce_sum(x, [3])
                log_det = lambda x: tf.reduce_sum(tf.log(tf.maximum(x, 1e-8)), [3])

                if params.kl_div_sig_mult > 0:
                    kl_div_weight = 2.0 * (tf.sigmoid(params.kl_div_sig_mult * tf.cast(global_step, tf.float32))-.5) * params.kl_div_weight
                else:
                    kl_div_weight = params.kl_div_weight


                for i in range(len(im_pyr)):
                    kl_div_loss = tf.reduce_sum(.5 * (trace(covariance[i]) + tf.reduce_sum(tf.square(mean[i]),[-1]) - params.code_size[i] - log_det(covariance[i])), [1,2])
                    self.losses.add_loss(tf.reduce_mean(kl_div_loss), 'kl_div_%d'%(i), kl_div_weight)

                    loss = tf.reduce_sum(tf.square(self.out_pyr_images[i] - im_pyr[i]), [1,2,3])
                    self.losses.add_loss(tf.reduce_mean(loss), 'regression_pyr_%d'%(i), params.regression_weight)

                regression_loss = tf.reduce_sum(tf.square(self.out_images - log_im) * self.pl['mask'], [1,2,3])
                self.losses.add_loss(tf.reduce_mean(regression_loss), 'regression', params.full_regression_weight)

                total_variation = variance_estimate.image_prior(self.out_images, self.pl['mask'])
                self.losses.add_loss(total_variation, 'total_variation', params.total_variation_weight)

                self.loss = self.losses.full_loss()
                self.losses.add_summary()

            with tf.variable_scope('optimizer'):
                lr = tf.train.exponential_decay(
                    params.init_lr, global_step,
                    params.lr_decay_steps, params.lr_decay, staircase=True)

                optimizer = tf.train.AdamOptimizer(lr)
                self.opt = optimizer.minimize(
                    self.loss,
                    var_list=var_collect.collect_scope('VAE', self.graph),
                    global_step=global_step)

                self.numerics_check = tf.add_check_numerics_ops()
                self.summaries = self.losses.get_summaries()

                if not params.normalize_images:
                    self.out_image = self.out_images
                else:
                    self.out_image = YCgCo_to_rgb(self.out_images)

        tf.initialize_variables(
            var_list=var_collect.collect_all(self.graph)).run(
                session=self.sess)

        self.saver = tf.train.Saver(var_collect.collect_scope('VAE',
                                                              self.graph))

        self.summ_writer = summary_writer(params.summary_output)
        self.summ_writer.add_writer('train', graph=self.graph)
        self.summ_writer.add_writer('val', graph=self.graph)

    def save(self, name):
        self.saver.save(self.sess, name)

    def restore(self, name):
        self.saver.restore(self.sess, name)

    def minimize(self, image, mask, epsilon):
        loss, summaries = self.sess.run(
            [self.loss, self.summaries, self.opt, self.numerics_check],
            self.pl.get_feed_dict({'phase_train': True,
                                   'epsilon':epsilon,
                                   'image': image,
                                   'mask': mask}))[0:2]

        self.summ_writer.add_summary('train', summaries)
        return loss

    def generate_predictions(self, image, mask, epsilon):
        im = self.sess.run([self.out_image],
            self.pl.get_feed_dict({'phase_train': False,
                                   'epsilon':epsilon,
                                   'image': image,
                                   'mask': mask}))
        return im[0]

class GMM:
    def __init__(self, codes_shape, scipy_gmm):
        self.graph = tf.Graph()
        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        self.sess = tf.Session(graph=self.graph, config=sess_config)

        print self.graph
        n,c = codes_shape

        with self.graph.as_default():
            self.pl = PlaceholderManager()
            self.pl.add_placeholder('codes', tf.float32, codes_shape)
            self.pl.add_placeholder('phase_train', tf.bool, [])

            with tf.variable_scope('GMM'):
                self.gmm = models.GMM(self.pl['codes'], scipy_gmm, self.pl['phase_train'])

        print var_collect.collect_all(self.graph)
        #tf.initialize_variables(
        #    var_list=var_collect.collect_all(self.graph)).run(session=self.sess)

    def score(self, codes):
        return self.sess.run([self.gmm.log_prob()],
                self.pl.get_feed_dict({'codes':codes}))

