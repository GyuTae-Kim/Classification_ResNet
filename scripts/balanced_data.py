import os
from glob import glob

import cv2

from sklearn.utils import shuffle

from utils.general import find_class, append_path


def load_data(train_base, val_base):
    cls = find_class(train_base)
    train_paths = _find_files(train_base, cls)
    val_paths = _find_files(val_base, cls)
    
    return train_paths, val_paths

def _find_files(basepath, cls):
    rst = []
    count = []
    
    for c in sorted(cls):
        files = glob(os.path.join(basepath, c, '*'))
        count += [len(files)]
    minimum = min(count)
    
    for c in sorted(cls):
        files = shuffle(glob(os.path.join(basepath, c, '*')), random_state=1024)[:minimum]
        append_files = append_path(basepath, c)(files)
        for p in append_files[minimum:]:
            os.remove(p)
        rst += append_files.tolist()
    
    return rst


if __name__ == '__main__':
    train_base = '../k-fashion/training'
    val_base = '../k-fashion/validation'
    
    train_paths = load_data(train_base)
    val_paths = load_data(val_base)
    
    
