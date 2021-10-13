import os
from glob import glob

import tensorflow as tf

import numpy as np

from utils.general import find_class, decode_jpg, append_path, make_labels, augment_layer, augment_using_layers


def load_data(configs):
    train_base = os.path.join(configs['param']['datapath'], 'training')
    val_base = os.path.join(configs['param']['datapath'], 'validation')
    cls = find_class(train_base)
    configs['param']['n_cls'] = len(cls)
    
    train_paths, count_train_labels = _find_files(train_base, cls)
    train_labels = make_labels(count_train_labels)
    val_files, count_val_labels = _find_files(val_base, cls)
    val_labels = make_labels(count_val_labels)
    
    return train_paths, train_labels, val_files, val_labels

def _find_files(basepath, cls):
    count_labels = []
    rst = []
    
    for c in sorted(cls):
        files = glob(os.path.join(basepath, c, '*'))
        count_labels += [len(files)]
        append_files = append_path(basepath, c)(files)
        rst += append_files.tolist()
    
    return rst, count_labels

def make_ds(configs):
    decoder = decode_jpg()
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
        .shuffle(1024)
        .prefetch(AUTO)
    )
    val_ds = tf.data.Dataset.from_tensor_slices((val_files, val_labels))
    val_ds = (
        val_ds
        .map(decoder)
        .batch(configs['param']['batch_size'])
        .cache()
        .shuffle(1024)
        .prefetch(AUTO)
    )
    
    return train_ds, train_labels, val_ds, val_labels

def make_test_ds(configs):
    decoder = decode_jpg()
    AUTO = tf.data.experimental.AUTOTUNE
    
    _, _, test_files, test_labels = load_data(configs)
    test_labels = tf.one_hot(test_labels, depth=configs['param']['n_cls'])
    test_ds = tf.data.Dataset.from_tensor_slices(test_files, test_labels)
    test_ds = (
        test_ds
        .map(decoder)
        .batch(configs['param']['batch_size'])
        .cache()
        .shuffle(1024)
        .prefetch(AUTO)
    )
    
    return test_ds, tf.one_hot(test_labels, depth=configs['param']['n_cls'])
