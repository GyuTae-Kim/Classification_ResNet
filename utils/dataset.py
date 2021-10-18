import os
from glob import glob

import tensorflow as tf
from sklearn.utils import shuffle

import numpy as np

from utils.general import find_class, decode_jpg, append_path, make_labels, augment_layer, augment_using_layers


def load_data(configs):
    train_base = os.path.join(configs['param']['datapath'], 'training')
    val_base = os.path.join(configs['param']['datapath'], 'validation')
    cls = find_class(train_base)
    configs['param']['cls'] = cls
    configs['param']['n_cls'] = len(cls)
    
    train_paths, count_train_labels = _find_files(train_base, cls)
    train_labels = make_labels(count_train_labels)
    train_paths, train_labels = shuffle(train_paths, train_labels, random_state=1024)
    val_paths, count_val_labels = _find_files(val_base, cls)
    val_labels = make_labels(count_val_labels)
    val_paths, val_labels = shuffle(val_paths, val_labels, random_state=1024)
    
    return train_paths, train_labels, val_paths, val_labels

def _find_files(basepath, cls):
    count_labels = []
    rst = []
    count = []
    
    for c in sorted(cls):
        files = glob(os.path.join(basepath, c, '*'))
        count += [len(files)]
    minimum = min(count)
    
    for c in sorted(cls):
        files = shuffle(glob(os.path.join(basepath, c, '*')), random_state=1024)[:minimum]
        count_labels += [len(files)]
        append_files = append_path(basepath, c)(files)
        del files
        rst += append_files.tolist()
    
    return rst, count_labels

def make_ds(configs):
    decoder = decode_jpg(image_size=configs['model_param']['input_shape'][:2])
    aug = augment_layer(configs['augment'])
    wrap_aug = augment_using_layers(aug)
    AUTO = tf.data.experimental.AUTOTUNE
    
    train_path, train_labels, val_files, val_labels = load_data(configs)
    train_labels = tf.one_hot(train_labels, depth=configs['param']['n_cls'])
    val_labels = tf.one_hot(val_labels, depth=configs['param']['n_cls'])
    train_ds = tf.data.Dataset.from_tensor_slices((train_path, train_labels))
    train_ds = (
        train_ds
        .map(decoder)
        .batch(configs['param']['batch_size'])
        .map(wrap_aug)
        .cache()
        .prefetch(AUTO)
    )
    val_ds = tf.data.Dataset.from_tensor_slices((val_files, val_labels))
    val_ds = (
        val_ds
        .map(decoder)
        .batch(configs['param']['batch_size'])
        .cache()
        .prefetch(AUTO)
    )
    
    return train_ds, train_labels, val_ds, val_labels

def make_test_ds(configs):
    decoder = decode_jpg(image_size=configs['model_param']['input_shape'][:2])
    AUTO = tf.data.experimental.AUTOTUNE
    
    _, _, test_files, test_labels = load_data(configs)
    test_labels = tf.one_hot(test_labels, depth=configs['param']['n_cls'])
    test_ds = tf.data.Dataset.from_tensor_slices((test_files, test_labels))
    test_ds = (
        test_ds
        .map(decoder)
        .batch(configs['param']['batch_size'])
    )
    
    return test_ds, test_labels
