import os
from glob import glob

import cv2

from sklearn.utils import shuffle

from utils.human_parse.handler import Handler as ParseHandler
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
        append_files = append_path(basepath, c)(files)[:minimum]
        del files
        rst += append_files.tolist()
    
    return rst

if __name__ == '__main__':
    train_base = 'E:/k-fashion/training'
    val_base = 'E:/k-fashion/validation'
    save_path = 'E:/k-fashion-parse'
    
    train_save = os.path.join(save_path, 'training')
    val_save = os.path.join(save_path, 'validation')
    train_paths, val_paths = load_data(train_base, val_base)
    
    handler = ParseHandler()
    
    for p in train_paths:
        filename = os.path.basename(p)
        origin = cv2.imread(p, cv2.IMREAD_COLOR)
        parse_input = cv2.resize(origin, dsize=(192, 256), interpolation=cv2.INTER_AREA)
        origin = cv2.resize(origin, dsize=(256, 256), interpolation=cv2.INTER_AREA)
        parse = handler(parse_input)
        origin *= parse
        cv2.imwrite(os.path.join(train_save, p), origin)
    
    for p in val_paths:
        filename = os.path.basename(p)
        origin = cv2.imread(p, cv2.IMREAD_COLOR)
        parse_input = cv2.resize(origin, dsize=(192, 256), interpolation=cv2.INTER_AREA)
        origin = cv2.resize(origin, dsize=(256, 256), interpolation=cv2.INTER_AREA)
        parse = handler(parse_input)
        origin *= parse
        cv2.imwrite(os.path.join(val_save, p), origin)
