import os
from glob import glob

import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import numpy as np
import math


def decode_jpg(image_size=(800, 800)):
    @tf.function
    def wrapper(file, label=None, image_size=image_size):
        bits = tf.io.read_file(file)
        image = tf.image.decode_jpeg(bits, channels=3)
        image = tf.cast(image, tf.float32) / 127.5 - 1.
        image = tf.image.resize(image, image_size)
        
        if label is None:
            return image
        else:
            return image, label
    
    return wrapper

def augment_layer(augment_dict):
    auglist = []
    if augment_dict['random_flip']:
        auglist += [preprocessing.RandomFlip('horizontal_and_vertical')]
    auglist += [
        preprocessing.RandomRotation(augment_dict['rotation_range']),
        preprocessing.RandomZoom(augment_dict['zoom_range'], augment_dict['zoom_range'])
    ]
    aug = tf.keras.Sequential(auglist)
    
    return aug

def augment_using_layers(aug):
    @tf.function
    def wrapper(x, y=None, aug=aug):
        x = aug(x)
        
        if y is None:
            return x
        else:
            return x, y
    
    return wrapper

def find_class(basepath):
    candidate = glob(os.path.join(basepath, '*'))
    category = []
    
    for c in candidate:
        if os.path.isdir(c):
            category += [os.path.basename(c)]
    
    return category

def append_path(base, pre):
    return np.vectorize(lambda file: os.path.join(base, pre, file))

def make_labels(data_len):
    labels = []
    
    for i, d in enumerate(data_len):
        labels += [i] * d
    
    return np.array(labels)

def calculate_confusion_matrix(submission, n_cls):
    pred = submission['pred']
    target = submission['target']
    confmat = np.zeros((n_cls, n_cls), dtype=np.int)
    
    for p, t in zip(pred, target):
        confmat[t, p] += 1
    
    return confmat

def mkdir(path):
    if os.path.exists(path):
        if not os.isdir(os.path.exists):
            os.mkdir(path)
    else:
        os.mkdir(path)

def remove_all(path):
    if os.path.exists(path):
        for f in os.scandir(path):
            os.remove(f.path)

def cosine_lr_decay(epoch, lrf):
    return lambda e, lr: lr * (((1. - math.cos(e * math.pi / epoch)) / 2) * (lrf - 1.) + 1.)

def linear_lr_decay(epoch, lrf):
    return lambda e, lr: lr * ((1 - (e / (epoch - 1))) * (1 - lrf) + 1)

def check_run_path():
    run_dir = './runs'
    listdir = glob(os.path.join(run_dir, 'run-*'))
    
    if len(listdir) == 0:
        path = os.path.join(run_dir, 'run-0')
        mkdir(path)
    else:
        dir_num = [int(d.split('-')[-1]) for d in listdir]
        cur_num = int(np.max(dir_num) + 1)
        path = os.path.join(run_dir, f'run-{cur_num}')
        mkdir(path)
    
    return path


class AugmentFlow(object):
    
    def __init__(self, augment_dict):
        self.img_gen = ImageDataGenerator(
            featurewise_center=augment_dict['featurewise_center'],
            featurewise_std_normalization=augment_dict['samplewise_center'],
            rotation_range=augment_dict['rotation_range'],
            width_shift_range=augment_dict['width_shift_range'],
            height_shift_range=augment_dict['height_shift_range'],
            zoom_range=augment_dict['zoom_range'],
            horizontal_flip=augment_dict['horizontal_flip'],
            vertical_flip=augment_dict['vertical_flip']
        )
    
    def convert(self, img):
        x = self.img_gen.random_transform(img)
        
        return x
