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


class BestSaverCallback(Callback):
    
    def __init__(self, checkpoint_path=None, check='val_loss'):
        super(BestSaverCallback, self).__init__()
        
        self.check = check
        self.checkpoint_path = checkpoint_path + 'bestcp-{epoch:04d}.ckpt'
        self.checkpoint_dir = os.path.dirname(self.checkpoint_path)
        self.best = np.INF
    
    def on_train_begin(self, logs=None):
        self.best = np.INF
        mkdir(self.checkpoint_dir)

    def on_train_end(self, logs=None):
        pass

    def on_epoch_begin(self, epoch, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.check)
        if np.less(current, self.best):
            self.best = current
            path = self.checkpoint_path.format(epoch=epoch)
            remove_all(self.checkpoint_dir)
            self.model.save_weights(path)
            print('Epoch {epoch:05d}: saving model to {path}'.format(epoch=epoch, path=path))

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
