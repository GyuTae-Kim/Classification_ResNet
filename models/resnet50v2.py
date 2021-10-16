import tensorflow as tf
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras import layers
from tensorflow.keras import Model
import tensorflow.keras.backend as K
from tensorflow.python.keras.utils import data_utils

from .common import stack_fn

"""
WEIGHTS_HASH = ('6343647c601c52e1368623803854d971',
                'c0ed64b8031c3730f411d2eb4eea35b5')
"""
WEIGHTS_HASH = ('3ef43a0b657b3be2300d5770ece849e0',
                   'fac2f116257151a9d068a22e544a4917')
BASE_WEIGHTS_PATH = (
    'https://storage.googleapis.com/tensorflow/keras-applications/resnet/')


def feature_extract(input_shape=None,
                    weights='imagenet',
                    use_bias=True,
                    pooling=None,
                    epsilon=1.001e-5,
                    model_name='resnet50v2'):
    """Instantiates the ResNet101V2 architecture.

    Args:
        input_shape (tuple): It should have exactly 3 inputs channels. Defaults to None.
        weights (str): one of `None` (random initialization),
         'imagenet' (pre-training on ImageNet),
          or the path to the weights file to be loaded. Defaults to 'imagenet'.
        use_bias (bool): Whether to use biases for convolutional layers or not. Defaults to False.
        pooling (str): Optional pooling mode for feature extraction
          - `None` means that the output of the model will be
              the 4D tensor output of the
              last convolutional layer.
          - `avg` means that global average pooling
              will be applied to the output of the
              last convolutional layer, and thus
              the output of the model will be a 2D tensor.
          - `max` means that global max pooling will
              be applied. Defaults to None.
        epsilon (float): Set epsilon of batch normalization layer. Defaults to 1.001e-5.
        model_name (str): Model name. Defaults to None.
    
    Returns:
        An Input layer and a `keras.Model` instance.
    """
    
    img_input = layers.Input(shape=input_shape)
    
    x = layers.ZeroPadding2D(padding=((3, 3), (3, 3)),
                             name='conv1_pad')(img_input)
    x = layers.Conv2D(filters=64,
                      kernel_size=7,
                      strides=2,
                      use_bias=use_bias,
                      kernel_regularizer='l2',
                      name='conv1_conv')(x)
    x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)),
                             name='pool1_pad')(x)
    x = layers.MaxPooling2D(pool_size=3,
                            strides=2,
                            name='pool1_pool')(x)
    
    x = stack_fn(x)
    
    x = layers.BatchNormalization(epsilon=epsilon,
                                  name='post_bn')(x)
    x = layers.Activation('relu', name='post_relu')(x)
    
    if pooling == 'avg':
        x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
    elif pooling == 'max':
        x = layers.GlobalMaxPooling2D(name='max_pool')(x)
    
    model = Model(inputs=img_input,
                  outputs=x,
                  name=model_name)
    
    if weights == 'imagenet':
        file_name = model_name + '_weights_tf_dim_ordering_tf_kernels_notop.h5'
        file_hash = WEIGHTS_HASH[1]
        weights_path = data_utils.get_file(file_name,
                                           BASE_WEIGHTS_PATH + file_name,
                                           cache_subdir='models',
                                           file_hash=file_hash)
        model.load_weights(weights_path)
    
    return model

def classification(n_cls):
    m = tf.keras.Sequential([
        layers.Dense(512, activation='relu', kernel_regularizer='l2', name='classification_1'),
        layers.Dense(n_cls, name='classification_2'),
        layers.Activation('softmax', dtype=tf.float32, name='predictions')
    ])
    
    return m


class ResNet50V2(Model):
    
    def __init__(self, configs):
        super(ResNet50V2, self).__init__()
        
        self.configs = configs
        
        self.base_layer = feature_extract(configs['model_param']['input_shape'],
                                          pooling=configs['model_param']['pooling'])
        self.base_layer.summary()
        self.classfier = classification(configs['param']['n_cls'])
        pass
        
    def call(self, inputs):
        x = self.base_layer(inputs)
        x = self.classfier(x)
        
        return x


if __name__ == "__main__":
    m = ResNet50V2(None)
    pass
