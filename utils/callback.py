import os

from tensorflow.keras.callbacks import Callback
import tensorflow.keras.backend as K

import numpy as np

from utils.general import mkdir, remove_all


class LRCallback(Callback):
    
    def __init__(self, schedule, logger):
        super(LRCallback, self).__init__()
        
        self.schedule = schedule
        self.logger = logger
        
    def on_train_begin(self, logs=None):
        lr = float(K.get_value(self.model.optimizer.learning_rate))
        self.logger.save_lr_log(lr)

    def on_train_end(self, logs=None):
        pass

    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, 'lr'):
            raise ValueError('Optimizer must have a "lr" attribute.')
        # Get the current learning rate from models's optimizer.
        lr = float(K.get_value(self.model.optimizer.learning_rate))
        # Call schedule function to get the scheduled learning rate.
        scheduled_lr = self.schedule(epoch, lr)
        # Set the value back to the optimizer before this epoch starts.
        K.set_value(self.model.optimizer.lr, scheduled_lr)
        self.logger.save_lr_log(scheduled_lr)

    def on_epoch_end(self, epoch, logs=None):
        pass

    def on_test_begin(self, logs=None):
        pass

    def on_test_end(self, logs=None):
        pass

    def on_predict_begin(self, logs=None):
        pass

    def on_predict_end(self, logs=None):
        pass

    def on_train_batch_begin(self, batch, logs=None):
        pass

    def on_train_batch_end(self, batch, logs=None):
        pass

    def on_test_batch_begin(self, batch, logs=None):
        pass

    def on_test_batch_end(self, batch, logs=None):
        pass

    def on_predict_batch_begin(self, batch, logs=None):
        pass

    def on_predict_batch_end(self, batch, logs=None):
        pass


class ModelSaverCallback(Callback):
    
    def __init__(self, save_freq=10, checkpoint_path=None):
        super(ModelSaverCallback, self).__init__()
        
        self.save_freq = save_freq
        self.best_checkpoint_dir = os.path.join(checkpoint_path, 'best')
        self.best_checkpoint_path = os.path.join(self.best_checkpoint_dir, 'bestcp-{epoch:04d}.ckpt')
        self.epoch_checkpoint_dir = os.path.join(checkpoint_path, 'epoch')
        self.epoch_checkpoint_path = os.path.join(self.epoch_checkpoint_dir, 'cp-{epoch:04d}.ckpt')
        
        self.step = 0
        self.best = np.INF
    
    def on_train_begin(self, logs=None):
        self.step = 0
        self.best = np.INF
        mkdir(self.epoch_checkpoint_dir)
        mkdir(self.best_checkpoint_dir)

    def on_train_end(self, logs=None):
        pass

    def on_epoch_begin(self, epoch, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):
        cur_loss = logs.get('val_loss')
        if np.less(cur_loss, self.best):
            self.best = cur_loss
            path = self.best_checkpoint_path.format(epoch=epoch)
            remove_all(self.best_checkpoint_dir)
            self.model.save_weights(path)
            print('Epoch {epoch:04d}: saving best model to {path}'.format(epoch=epoch, path=path))
        
        self.step += 1
        if self.step >= self.save_freq:
            self.step = 0
            path = self.epoch_checkpoint_path.format(epoch=epoch)
            self.model.save_weights(path)
            print('Epoch {epoch:04d}: saving model to {path}'.format(epoch=epoch, path=path))

    def on_test_begin(self, logs=None):
        pass

    def on_test_end(self, logs=None):
        pass

    def on_predict_begin(self, logs=None):
        pass

    def on_predict_end(self, logs=None):
        pass

    def on_train_batch_begin(self, batch, logs=None):
        pass

    def on_train_batch_end(self, batch, logs=None):
        pass

    def on_test_batch_begin(self, batch, logs=None):
        pass

    def on_test_batch_end(self, batch, logs=None):
        pass

    def on_predict_batch_begin(self, batch, logs=None):
        pass

    def on_predict_batch_end(self, batch, logs=None):
        pass
