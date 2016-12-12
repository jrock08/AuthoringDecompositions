import os
import random
import numpy as np
import scipy.signal
import math
import threading

from aiutils.data import batch_creators
from aiutils.vis.image_io import imread

class Image(object):
    def __init__(self, image, mask, name):
        self.image = image
        self.mask = mask
        self.crop_mask = self.mask
        self.name = name
        self.mask_crop_size = [1,1]

    def compute_filt_mask(self, imsize):
        if imsize == self.mask_crop_size:
            return

        self.mask_crop_size = imsize

        filt = np.full(imsize, 1, dtype='float')
        self.crop_mask = scipy.signal.fftconvolve(self.mask, filt, mode='same')

    def get_crop(self, imsize, sample_quality=.9):
        loc = self.get_loc(imsize, sample_quality=sample_quality)
        return self.crop_image(loc, imsize)

    def get_scale_crop(self, imsize, sample_quality=.9, scale_val=2):
        loc = self.get_loc(imsize, sample_quality=sample_quality)
        return self.scale_crop_image(loc, imsize, scale_val)

    def get_rot_crop(self, imsize, sample_quality=.9, angle=10):
        loc = self.get_loc(imsize, sample_quality=sample_quality)
        return self.rot_crop_image(loc, imsize, angle)

    def get_rot_scale_crop(self, imsize, sample_quality=.9, angle=10, scale_val=2):
        loc = self.get_loc(imsize, sample_quality=sample_quality)
        return self.rot_scale_crop_image(loc, imsize, angle, scale_val)

    def rot_scale_crop_image(self, loc, imsize, angle, scale_val):
        assert scale_val > 0 and scale_val < 3
        new_imsize = [int(math.ceil(scale_val * float(x)) + 2) for x in imsize]
        imsize_diff = [x - y for x, y in zip(new_imsize, imsize)]
        imsize_pre = [int(math.floor(x / 2)) for x in imsize_diff]

        new_loc = [x - y for x,y in zip(loc, imsize_pre)]

        if angle != 0:
            im = self.rot_crop_image(new_loc, new_imsize, angle)
        else:
            im = self.crop_image(new_loc, new_imsize)

        scale_vals = [float(x)/y for x,y in zip(imsize,new_imsize)]
        im = im.scale_image_size(scale_vals)

        # crop will no-op if the image is correctly size, but crop or pad if it's not
        im = im.crop_image([0,0], imsize)

        return im

    def scale_crop_image(self, loc, imsize, scale_val):
        assert scale_val > 0 and scale_val < 3
        new_imsize = [int(math.ceil(scale_val * float(x))) for x in imsize]
        imsize_diff = [x - y for x, y in zip(new_imsize, imsize)]
        imsize_pre = [int(math.floor(x / 2)) for x in imsize_diff]

        new_loc = [x - y for x,y in zip(loc, imsize_pre)]

        im = self.crop_image(new_loc, new_imsize)
        scale_vals = [float(x)/y for x,y in zip(imsize,new_imsize)]
        im = im.scale_image_size(scale_vals)

        # crop will no-op if the image is correctly size, but crop or pad if it's not
        im = im.crop_image([0,0], imsize)

        return im

    def rot_crop_image(self, loc, imsize, angle):
        assert angle >= 0 and angle <= 45
        ang = math.cos(math.radians(45-angle)) * math.sqrt(2.0)
        #new_imsize = [math.ceil(x) for x in math.cos(math.radians(45-angle)) * math.sqrt(2.0) * imsize]
        new_imsize = [int(math.ceil(ang * float(x)) + 2) for x in imsize]

        imsize_diff = [x - y for x, y in zip(new_imsize, imsize)]
        imsize_pre = [int(math.floor(x / 2)) for x in imsize_diff]

        new_loc = [x - y for x,y in zip(loc, imsize_pre)]

        im = self.crop_image(new_loc, new_imsize)
        im = im.rotate_image(angle)
        im = im.crop_image(imsize_pre, imsize)

        return im


    def scale_image_size(self, scale_val):
        im = Image(scipy.ndimage.zoom(self.image, scale_val + [1], order=1),
                scipy.ndimage.zoom(self.mask, scale_val, order=0) > .5,
                self.name)

        return im


    def rotate_image(self, angle):
        im = Image(scipy.ndimage.interpolation.rotate(self.image, angle, order=1, reshape=False),
                scipy.ndimage.interpolation.rotate(self.mask, angle, order=0, reshape=False) > .5,
                self.name)

        return im

    def crop_image(self, loc, imsize):
        im = Image(Image.crop_im(self.image, loc, imsize),
                   Image.crop_im(self.mask, loc, imsize),
                   self.name)

        return im

    def get_loc(self, imsize, sample_quality=.9):
        self.compute_filt_mask(imsize)

        h,w = self.image.shape[0:2]
        X = np.arange(h*w)
        X_img = X.reshape(h,w)
        while np.sum(self.crop_mask > sample_quality*np.prod(imsize)) < 100:
            sample_quality*=.9
        loc = random.choice(X_img[self.crop_mask > sample_quality*np.prod(imsize)])

        h_, w_ = np.where(X_img == loc)
        h_ = max([0], h_ - imsize[0]/2)
        w_ = max([0], w_ - imsize[1]/2)

        loc = (h_[0],w_[0])
        return loc

    @staticmethod
    def crop_im(image, loc, imsize):
        h_ = loc[0]
        w_ = loc[1]

        im = image[h_:h_+imsize[0], w_:w_+imsize[1]]

        if im.shape[0] != imsize[0] or im.shape[1] != imsize[1]:
            if len(im.shape) == 3:
                im = np.pad(im, ((0,imsize[0]-im.shape[0]),(0,imsize[1]-im.shape[1]),(0,0)), 'constant')
            else:
                im = np.pad(im, ((0,imsize[0]-im.shape[0]),(0,imsize[1]-im.shape[1])), 'constant')

        return im

class ImageData(object):
    def __init__(self, image_root, mask_root,
            im_list = [], patch_size=[100,100], sample_quality=.95,
            color=False):

        if len(im_list) == 0:
            im_list = sorted(os.listdir(image_root))

        self.im_list = im_list
        self.image_root = image_root
        self.mask_root = mask_root

        self._patch_size = patch_size
        self._sample_quality = sample_quality
        self._color = color

        self._epochs_completed = 0
        self._index_in_epoch = 0

    def sample_to_image_id(self, sample):
        return self.im_list[sample]

    def get_image_obj(self, im_name):
        return Image(
                imread(self.image_root + im_name)/256.0,
                imread(self.mask_root + im_name) != 0,
                im_name)

    def ims(self, idx=0):
        return self.get_image_obj(self.im_list[idx])

    def get_data_single(self, sample):
        return self.get_unscaled(sample, self._patch_size, self._sample_quality, self._color)

    def get_data_threaded(self, sample, batch_list, worker_id):
        batch = self.get_data_single(sample)
        batch_list[worker_id] = batch

    def get_data(self, samples):
        batch_size = len(samples)
        batch_list = [None]*batch_size
        worker_ids = range(batch_size)
        workers = []
        for count, sample in enumerate(samples):
            worker = threading.Thread(
                    target = self.get_data_threaded,
                    args = (sample, batch_list, worker_ids[count]))
            worker.setDaemon(True)
            worker.start()
            workers.append(worker)

        for worker in workers:
            worker.join()

        batch = dict()
        batch['image'] = np.zeros(
                [batch_size] + self._patch_size + [3 if self._color else 1],
                dtype=np.float32)
        batch['mask'] = np.zeros(
                [batch_size] + self._patch_size + [1],
                dtype=np.float32)

        for i, single_sample in enumerate(batch_list):
            batch['image'][i,:,:,:] = single_sample.image
            batch['mask'][i,:,:,:] = np.expand_dims(single_sample.mask,-1)

        return batch

    def get_unscaled(self, sample, patch_size=[100,100], sample_quality=.95, color=False):
        return self.ims(sample).get_crop(patch_size, sample_quality)


    def get_scaled(self, sample, patch_size=[100,100], sample_quality=.95, color=False):
        truncnorm = lambda mu,var,lower,upper: scipy.stats.truncnorm.rvs((lower-mu) / var, (upper-mu)/var, loc=mu, scale=var, size=1)[0]

        angle = truncnorm(10, 10, 0, 45)
        scale = truncnorm(1, .5, .3, 1.2)
        return self.ims(idx).get_rot_scale_crop(patch_size, sample_quality, angle, scale)

def create_batch_generator(data_mgr, batch_size, queue_size, num_epochs, num_samples, same_process=False):

    index_generator = batch_creators.sequential(batch_size, num_samples, num_epochs)

    if same_process:
        batch_generator = batch_creators.batch_generator(
            data_mgr, index_generator)
    else:
        batch_generator = batch_creators.async_batch_generator(
            data_mgr, index_generator, queue_size)

    return batch_generator

def default_data(image_root, mask_root, patch_size, batch_size, queue_size, num_samples, color=True, same_process=True, num_epochs=None):
    data_mgr = ImageData(image_root, mask_root, patch_size=patch_size, color=color)
    num_epochs = num_epochs or -1

    batch_generator = create_batch_generator(data_mgr, batch_size, queue_size, num_epochs, num_samples, same_process)

    return data_mgr, batch_generator

