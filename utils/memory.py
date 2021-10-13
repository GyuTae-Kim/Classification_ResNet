import logging

import numpy as np


class Memory(object):
    
    def __init__(self):
        """
        self.logger = logging()
        """
        self.initialize()
        self.last_lr = np.Inf
    
    def initialize(self):
        """
        logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -  %(message)s',
                            datafmt='%m/%d/%Y %H:%M%S',
                            level=logging.INFO)
        """
        self.train_log = {'lr': []}
        
    def save_lr_log(self, lr):
        lr = float(lr)
        self.train_log['lr'] += [lr]
        if self.last_lr != lr:
            self.last_lr = lr
