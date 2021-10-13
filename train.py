import argparse
import yaml
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from models.resnet101v2 import ResNet101V2
from utils.dataset import make_ds
from utils.logger import *
from utils.visualize import *
from utils.callback import LRCallback, ModelSaverCallback
from options.train_option import TrainOptions


def train(configs):
    pass

def set_configs(configs):
    if configs['param']['continue_train_path']:
        configs_path = os.path.join(configs['param']['continuew_train_path'], 'configs.yaml')
        with open(configs_path, 'r') as f:
            new_configs = yaml.load(f, Loader=yaml.FullLoader)
        new_configs['param']['n_epochs'] = configs['param']['n_epochs']
        new_configs['param']['batch_size'] = configs['param']['batch_size']
        if configs['param']['use_best']:
            new_configs['param']['load_weights'] = os.path.join(configs['param']['continue_train_path'], 'best')
        else:
            new_configs['param']['load_weights'] = os.path.join(configs['param']['continue_train_path'], 'epoch')
        
        return new_configs
    
    else:
        configs['params']
        
        return configs

if __name__ == '__main__':
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    train_options = TrainOptions()
    opt = train_options.parse()
    with open(opt.model_params, 'r') as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)
    configs['param'] = vars(opt)
    configs = set_configs(configs)
    train_options.print_options(configs)
    
    train(configs)
