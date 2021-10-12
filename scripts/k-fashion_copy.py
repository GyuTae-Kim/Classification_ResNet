from glob import glob
from shutil import copyfile
from tqdm import tqdm
import os
import json
import argparse


CLS = ['outerwear', 'tops', 'dresses', 'pants']


def copy_files(imgs, label_path, dest):
    global CLS
    targets = []
    
    if not os.path.exists(dest):
        os.mkdir(dest)
    
    for c in CLS:
        target = os.path.join(dest, c)
        if not os.path.exists(target):
            os.mkdir(target)
            print(f'Create new dir: {target}')
        targets += [target]
    
    for i in tqdm(range(len(imgs))):
        subdir = imgs[i].split(os.sep)[-2]
        basename = os.path.basename(imgs[i])
        noext = os.path.splitext(basename)[0] + '.json'
        data = os.path.join(label_path, subdir, noext)
        with open(data, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        meta = json_data['metadata']
        
        if CLS[0] in meta.keys():
            copyfile(imgs[i], os.path.join(targets[0], basename))
        elif CLS[1] in meta.keys():
            copyfile(imgs[i], os.path.join(targets[1], basename))
        elif CLS[2] in meta.keys():
            copyfile(imgs[i], os.path.join(targets[2], basename))
        elif CLS[3] in meta.keys():
            copyfile(imgs[i], os.path.join(targets[3], basename))
        else:
            print('\nError:', str(imgs[i]), str(meta.keys()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type=str, required=True)
    parser.add_argument('--dest', type=str, required=True)
    args = parser.parse_args()
    
    train_img_path = os.path.join(args.src, 'training', 'data')
    train_label_path = os.path.join(args.src, 'training', 'labels')
    val_img_path = os.path.join(args.src, 'validation', 'data')
    val_label_path = os.path.join(args.src, 'validation', 'labels')
    
    if not os.path.exists(train_img_path):
        raise FileNotFoundError('Not found train directory: {path}'.format(path=train_img_path))
    if not os.path.exists(train_label_path):
        raise FileNotFoundError('Not found train directory: {path}'.format(path=train_label_path))
    if not os.path.exists(val_img_path):
        raise FileNotFoundError('Not found validation directory: {path}'.format(path=val_img_path))
    if not os.path.exists(val_label_path):
        raise FileNotFoundError('Not found validation directory: {path}'.format(path=val_label_path))
        
    train_imgs = glob(os.path.join(train_img_path, '*', '*.jpg'))
    print(f'Find train img: {len(train_imgs)}')
    val_imgs = glob(os.path.join(val_img_path, '*', '*.jpg'))
    print(f'Find validation img: {len(val_imgs)}')
    
    print('Copy train imgs..')
    copy_files(train_imgs, train_label_path, os.path.join(args.dest, 'training'))
    print('Copy validation imgs..')
    copy_files(val_imgs, val_label_path, os.path.join(args.dest, 'validation'))
