import yaml
import os
import warnings

warnings.simplefilter('ignore')
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras import metrics
from tensorflow.keras.mixed_precision import experimental as mixed_precision

policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_policy(policy)

import numpy as np

from models.resnet50v2 import ResNet50V2
from utils.dataset import make_ds, make_test_ds
from utils.memory import Memory
from utils.visualize import visual_confusion_matrix, visual_count_label, visual_f1score,\
    visual_learning_rate, visual_loss, visual_accuracy
from utils.callbacks import LRCallback, ModelSaverCallback
from utils.general import cosine_lr_decay, linear_lr_decay, check_run_path
from options.train_option import TrainOptions


def train(configs):
    # before training
    mem = Memory()
    train_ds, train_labels, val_ds, val_labels = make_ds(configs)
    n_cls = configs['param']['n_cls']
    visual_count_label(train_labels, val_labels, configs['param']['run_path'])
    
    # load optimizer
    if configs['optimizer']['adam']:
        opt = tf.keras.optimizers.Adam(learning_rate=configs['optimizer']['init_lr'])
    else:
        opt = tf.keras.optimizers.SGD(learning_rate=configs['optimizer']['init_lr'], momentum=configs['optimizer']['momentum'])
    opt = mixed_precision.LossScaleOptimizer(opt, loss_scale='dynamic')
    
    # load model
    model = ResNet50V2(configs)
    dummy = tf.random.normal((1, *configs['model_param']['input_shape']), dtype='float')
    model(dummy)
    del dummy
    model.compile(
        optimizer=opt,
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=[
            'accuracy',
            metrics.Precision(),
            *(metrics.Precision(name=f'precision{i}', class_id=i) for i in range(n_cls)),
            metrics.Recall(),
            *(metrics.Recall(name=f'recall{i}', class_id=i) for i in range(n_cls)),
        ]
    )
    model.summary()
    
    # callbacks
    if configs['optimizer']['cosine_decay']:
        lr_decay = cosine_lr_decay(configs['param']['n_epochs'], configs['optimizer']['lrf'])
    else:
        lr_decay = linear_lr_decay(configs['param']['n_epochs'], configs['optimizer']['lrf'])

    callbacks = [
        LRCallback(configs['optimizer']['init_lr'], lr_decay, mem),
        ModelSaverCallback(configs['param']['save_epoch_freq'],
                           configs['param']['run_path'])
    ]
    
    # train start
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        callbacks=callbacks,
        batch_size=configs['param']['batch_size'],
        epochs=configs['param']['n_epochs'],
        initial_epoch=0
    )
    
    # testing
    submission = dict()
    test_ds, submission['target'] = make_test_ds(configs)
    submission['pred'] = model.predict(test_ds)
    
    # visualize
    print('Test Matched {} / {}'.format(np.sum(submission['target'] == submission['pred']), submission['target'].shape[0]))
    visual_accuracy(history['accuracy'], history['val_accuracy'])
    visual_loss(history['loss'], history['val_loss'])
    visual_confusion_matrix(submission, configs['param']['run_path'])
    visual_learning_rate(mem.train_log['lr'], configs['param']['run_path'])
    visual_f1score(
        history['precision'],
        history['recall'],
        [history[f'precision{i}'] for i in range(n_cls)],
        [history[f'recall{i}'] for i in range(n_cls)],
        configs['param']['run_path']
    )

def set_configs(configs):
    if configs['param']['continue_train_path']:
        configs_path = os.path.join(configs['param']['continuew_train_path'], 'configs.yaml')
        with open(configs_path, 'r') as f:
            new_configs = yaml.load(f, Loader=yaml.FullLoader)
        new_configs['param'] = dict()
        new_configs['param']['n_epochs'] = configs['param']['n_epochs']
        new_configs['param']['batch_size'] = configs['param']['batch_size']
        new_configs['param']['run_path'] = configs['param']['continue_train_path']
        new_configs['param']['continue_train_path'] = configs['param']['continue_train_path']
        new_configs['param']['datapath'] = configs['param']['datapath']
        if configs['param']['use_best']:
            new_configs['param']['load_weights'] = os.path.join(configs['param']['continue_train_path'], 'best')
        else:
            new_configs['param']['load_weights'] = os.path.join(configs['param']['continue_train_path'], 'epoch')
        
        return new_configs
    
    else:
        configs['param']['load_weights'] = 'imagenet'
        configs['param']['run_path'] = check_run_path()
        
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
    print(f'Compute dtype: {policy.compute_dtype}')
    print(f'Variable dtype: {policy.variable_dtype}')
    
    train(configs)
