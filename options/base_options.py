import argparse
import os
import tensorflow as tf


class BaseOptions(object):
    """This class defines options used during both training and test time.
    
    It also implements several helper functions such as parsing, prining, and saving the options.
    It also gathers additional options defined in <modify_commandline_options> functions in both dataset class and model class.
    """

    def __init__(self):
        self.initialized = False
    
    def initialize(self, parser):
        # model parameters
        parser.add_argument('--batch_size', type=int, default=16, help='input batch size')
        parser.add_argument('--model_params', type=str, default='./data/resnet101v2.yaml', help='chooses which model params to use.')
        # network saving and loading parameters
        parser.add_argument('--use_best', action='store_true', help='whether uses best checkpoint or frequency saved checkpoint')
        
        self.initialized = True
        
        return parser
    
    def print_options(self, opt):
        """Print options.

        Args:
            opt (dict): which you want to print.
        """
        msg = ''
        msg += '----------------- Options ---------------\n'
        for k, v in sorted(opt.items()):
            msg += '{:>25}: {:<30}\n'.format(str(k), str(k))
        msg += '----------------- End -------------------'
        print(msg)
    
    def parse(self):
        if not self.initialized:
            parser = argparse.ArgumentParser()
            parser = self.initialize(parser)
        
        opt, _ = parser.parse_known_args()
        
        return opt
