from .base_options import BaseOptions


class TestOptions(BaseOptions):
    """This class includes testing options.
    
    It also includes shared options defined in BaseOptions.
    """
    
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        # network saving and loading parameters
        parser.add_argument('--run_path', type=str, default='', help='load network, setting and etc.. example) "./runs/exp01"')
        # testing parameters
        parser.add_argument('--data_path', type=int, default=10, help='frequency of saving checkpoints at the end of epochs')
        
        return parser
