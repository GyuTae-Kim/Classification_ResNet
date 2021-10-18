import yaml
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras import metrics
from tensorflow.keras.mixed_precision import experimental as mixed_precision

policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_policy(policy)

import numpy as np

from models.resnet50v2 import ResNet50V2
from utils.dataset import make_test_ds
from utils.visualize import visual_confusion_matrix, visual_count_label
from utils.general import check_run_path
from options.test_option import TestOptions


def test(configs):
    # before testing
    submission = dict()
    test_ds, submission['target'] = make_test_ds(configs)
    n_cls = configs['param']['n_cls']
    
    # load model
    model = ResNet50V2(configs)
    dummy = tf.random.normal((1, *configs['model_param']['input_shape']), dtype='float')
    model(dummy)
    del dummy
    latest = tf.train.latest_checkpoint(configs['param']['load_weights'])
    tf.train.Checkpoint.restore(latest).assert_consumed()
    model.load_weights(latest)
    model.compile(
        optimizer='sgd',
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=[
            'accuracy',
        ]
    )
    model.summary()
    
    # testing
    submission['pred'] = model.predict(test_ds)
    submission['pred'] = np.argmax(submission['pred'], 1)
    submission['target'] = np.argmax(submission['target'], 1)
    
    # visualize
    print('Test Matched {} / {}'.format(np.sum(submission['target'] == submission['pred']), submission['target'].shape[0]))
    visual_confusion_matrix(submission, n_cls, configs['param']['run_path'])

def set_configs(configs):
    configs['param']['batch_size'] = configs['param']['batch_size']
    if configs['param']['use_best']:
        configs['param']['load_weights'] = os.path.join(configs['param']['run_path'], 'best')
    else:
        configs['param']['load_weights'] = os.path.join(configs['param']['run_path'], 'epoch')
    
    return configs

if __name__ == '__main__':
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    test_options = TestOptions()
    opt = test_options.parse()
    with open(opt.model_params, 'r') as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)
    configs['param'] = vars(opt)
    configs = set_configs(configs)
    test_options.print_options(configs)
    print(f'Compute dtype: {policy.compute_dtype}')
    print(f'Variable dtype: {policy.variable_dtype}')
    
    test(configs)

