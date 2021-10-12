import os
from glob import glob

import tensorflow as tf

from utils.general import decode_jpg, append_path, make_labels, AugmentFlow


def load_data(config):
    train_base = os.path.join(config['param']['datapath'], 'training')
    val_base = os.path.join(config['param']['datapath'], 'validation')
    cls = config['param']['cls']
    
    train_paths, count_train_labels = _find_files(train_base, cls)
    train_labels = make_labels(count_train_labels)
    del count_train_labels
    val_files, count_val_labels = _find_files(val_base, cls)
    val_labels = make_labels(count_val_labels)
    del count_val_labels
    
    return train_paths, train_labels, val_files, val_labels

def _find_files(basepath, cls):
    count_labels = []
    rst = []
    
    for c in sorted(cls):
        files = glob(os.path.join(basepath, c, '*'))
        count_labels += [len(files)]
        append_files = append_path(basepath, c)(files)
        rst += append_files
    
    return rst, count_labels

def make_ds(config):
    aug = AugmentFlow(config['augment'])
    AUTO = tf.data.experimental.AUTOTUNE
    
    train_path, train_labels, val_files, val_labels = load_data(config)
    train_ds = tf.data.Dataset.from_tensor_slices((train_path, train_labels))
    train_ds = (
        train_ds
        .batch(config['param']['batch_size'])
        .map(lambda x, y: decode_jpg(x, y, augment=aug.convert), num_parallel_calls=AUTO)
        .cache()
        .repeat()
        .shuffle(1024)
        .prefetch(AUTO)
    )
    val_ds = tf.data.Dataset.from_tensor_slices((val_files, val_labels))
    val_ds = (
        val_ds
        .batch(config['param']['batch_size'])
        .map(lambda x, y: decode_jpg(x, y), num_parallel_calls=AUTO)
        .cache()
        .repeat()
        .shuffle(1024)
        .prefetch(AUTO)
    )
    
    return train_ds, val_ds
