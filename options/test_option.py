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
        
        
        return parser
