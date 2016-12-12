from aiutils.vis.image_io import imwrite

import cPickle as pickle

import numpy as np
import random

import os
import os.path

def select(ims):
    output = np.zeros_like(ims['image'], dtype=np.float32)
    labels = []
    names = ['image', 'albedo', 'shading']
    for i in range(ims['image'].shape[0]):
        l = random.choice(range(len(names)))
        output[i] = ims[names[l]][i]
        labels.append(l)

    return output, labels

def select_albedo_shading(ims):
    output = np.zeros_like(ims['image'], dtype=np.float32)
    labels = []
    names = ['albedo', 'shading']
    for i in range(ims['image'].shape[0]):
        l = random.choice(range(len(names)))
        output[i] = ims[names[l]][i]
        labels.append(l)

    return output, labels

def select_w_noise(ims, one_label=True):
    output = np.zeros_like(ims['image'], dtype=np.float32)
    labels = []
    masks = np.zeros_like(ims['mask'], dtype=np.float32)
    names = ['image', 'albedo', 'shading']
    for i in range(ims['image'].shape[0]):
        l = random.choice(range(len(names)))
        if random.choice([True, False]):
            output[i] = ims[names[l]][i]
            labels.append(l)
            masks[i] = ims['mask'][i]
        else:
            im = ims[names[l]][i]
            r = .1 * (np.random.random(im.shape)-.5)
            output[i] = np.minimum(np.maximum(ims[names[l]][i] + r, 0.0), 1.0)
            if one_label:
                labels.append(4)
            else:
                labels.append(len(names) + l)
    return output, labels, masks

def print_images(ims, name, out_dir):
    if ims.shape[0] == 1:
        imwrite((255 * ims[0]).astype(np.uint8),
            out_dir + name + '.png')
    else:
        for i in range(ims.shape[0]):
            imwrite((255 * ims[i]).astype(np.uint8),
                out_dir + name + '_%d.png'%(i))

def check_ims(ims, name, out_dir):
    checked = True
    if ims.shape[0] == 1:
        checked = checked and os.path.isfile(out_dir + name + '.png')
    else:
        for i in range(ims.shape[0]):
            checked = checked and os.path.isfile(out_dir + name + '_%d.png'%(i))
    return checked

def save_params(param, out_dir, name):
    print out_dir + '/' + name + '.pkl'
    with open(out_dir + '/' + name + '.pkl', 'w') as output:
        pickle.dump(param, output, -1)

def load_params(out_dir, name):
    print out_dir + '/' + name + '.pkl'
    with open(out_dir + '/' + name + '.pkl') as inp:
        return pickle.load(inp)
