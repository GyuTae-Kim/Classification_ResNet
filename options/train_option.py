from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    """This class includes training options.
    
    It also includes shared options defined in BaseOptions.
    """
    
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        # network saving and loading parameters
        parser.add_argument('--continue_train_path', type=str, default='', help='load network, setting and etc.. example) "./runs/exp01"')
        parser.add_argument('--save_epoch_freq', type=int, default=10, help='frequency of saving checkpoints at the end of epochs')
        # training parameters
        parser.add_argument('--n_epochs', type=int, default=50, help='number of epochs with the initial learning rate')
        
        return parser
