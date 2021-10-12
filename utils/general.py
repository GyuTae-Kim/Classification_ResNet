import os
from glob import glob

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import numpy as np


def decode_jpg(file, label=None, image_size=(800, 800), depth=None, aug_func=None):
    bits = tf.io.read_file(file)
    image = tf.image.decode_jpeg(bits, channels=3)
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, image_size)
    
    if aug_func:
        image = aug_func(image)
        
    if label is None:
        return image
    else:
        label = tf.one_hot(label, depth=depth)
        return image, label

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

def cosine_lr_decay(epoch, lr):
    pass


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
