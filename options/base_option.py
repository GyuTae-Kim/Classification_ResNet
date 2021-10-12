import argparse
import os
import tensorflow as tf


class BaseOption():
    """This class defines options used during both training and test time.
    
    It also implements several helper functions such as parsing, prining, and saving the options.
    It also gathers additional options defined in <modify_commandline_options> functions in both dataset class and model class.
    """

    def __init__(self):
        self.initialized = False
    
    def initialize(self, parser):
        parser.add_argument('--')
