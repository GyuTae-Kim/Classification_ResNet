import numpy as np


class Logger(object):
    
    def __init__(self):
        self.initialize()
        self.last_lr = np.INF
    
    def initialize(self):
        self.train_log = {'lr': []}
        self.test_log = {
            'pred': None,
            'target': None
        }
        self.data_log = {'n_cat': None}
        
    def save_lr_log(self, epoch, lr):
        lr = float(lr)
        self.train_log['lr'] += [lr]
        if self.last_lr != lr:
            self.last_lr = lr
            print('Epoch {epoch:05d}: set learning rate {lr}'.format(epoch=epoch, lr=lr))
    
    def save_test_log(self, pred, target):
        self.test_log['pred'] = pred
        self.test_log['target'] = target
