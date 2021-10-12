from tensorflow.keras import layers


def Conv(x, filters, kernel_size=3, stride=1, epsilon=1.001e-5, use_bias=False, name=None):
    """A convolution layer.

    Args:
        x (tensor): Input tensor.
        filters (int): Filters of the convolution layer.
        kernel_size (int): Kernel size of the convolution layer. Defaults to 3.
        stride (int): Stride of the convolution layer. Defaults to 1.
        epsilon (float): Set epsilon of batch normalization layer. Defaults to 1.001e-5.
        use_bias (bool): Use bias in convolution layer if True. Defaults to False.
        name (str): Block label. Defaults to None.
    
    Returns:
        Output tensor for the convolutional layer.
    """
    x = layers.Conv2D(filters=filters,
                      kernel_size=kernel_size,
                      strides=stride,
                      use_bias=use_bias,
                      kernel_regularizer='l2',
                      name=name + '_conv')(x)
    x = layers.BatchNormalization(epsilon=epsilon,
                                  name=name + '_bn')(x)
    x = layers.Activation('relu',
                          name=name + '_relu')(x)
    
    return x


def res_block(x, filters, kernel_size=3, stride=1, epsilon=1.001e-5, conv_shortcut=False, name=None):
    """A residual block.

    Args:
        x (tensor): Input tensor.
        filters (int): Filters of the bottleneck layer.
        kernel_size (int): Kernel size of the bottleneck layer. Defaults to 3.
        stride (int): Stride of the first layer. Defaults to 1.
        epsilon (float): Set epsilon of batch normalization layer. Defaults to 1.001e-5.
        conv_shortcut (bool): Use convolution shortcut if True,
          otherwise identity shortcut. Defaults to False.
        name (str): Block label. Defaults to None.
    
    Returns:
        Output tensor for the residual block.
    """
    x = layers.BatchNormalization(epsilon=epsilon,
                                  name=name + '_preact_bn')(x)
    x = layers.Activation('relu',
                          name=name + '_preact_relu')(x)
    
    if conv_shortcut:
        shortcut = layers.Conv2D(filters=4 * filters,
                                 kernel_size=1,
                                 strides=stride,
                                 kernel_regularizer='l2',
                                 name=name + '_0_conv')(x)
    else:
        shortcut = layers.MaxPool2D(pool_size=1,
                                    strides=stride)(x) if stride > 1 else x
    
    x = Conv(x,
             filters=filters,
             kernel_size=1,
             strides=1,
             use_bias=False,
             name=name + '_1')
    
    x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name=name + '_2_pad')(x)
    x = Conv(x,
             filters=filters,
             kernel_size=kernel_size,
             strides=stride,
             use_bias=False,
             name=name + '_2')
    
    x = layers.Conv2D(filters=4 * filters,
                      kernel_size=1,
                      kernel_regularizer='l2',
                      name=name + '_3_conv')(x)
    x = layers.Add(name=name + '_out')([shortcut, x])
    
    return x


def res_stack(x, filters, blocks, stride=2, epsilon=1.001e-5, name=None):
    """A set of stacked of residual blocks.

    Args:
        x (tensor): Input tensor.
        filters (int): Filters of the bottleneck layer in a block.
        blocks (int): Blocks in the stacked blocks.
        stride (int): Stride of the first layer in the first block. Defaults to 2.
        epsilon (float): Set epsilon of batch normalization layer. Defaults to 1.001e-5.
        name (str): Stack label. Defaults to None.
    
    Returns:
        Output tensor for the stacked block.
    """
    x = res_block(x=x,
                  filters=filters,
                  epsilon=epsilon,
                  conv_short_cut=True,
                  name=name + '_block1')
    for i in range(2, blocks):
        x = res_block(x=x,
                      filters=filters,
                      epsilon=epsilon,
                      name=name + '_block' + str(i))
    x = res_block(x=x,
                  filters=filters,
                  stride=stride,
                  epsilon=epsilon,
                  name=name + '_block' + str(blocks))
    
    return x


def stack_fn(x):
    x = res_stack(x, 64, 3, name='conv2')
    x = res_stack(x, 128, 4, name='conv3')
    x = res_stack(x, 256, 23, name='conv4')
    x = res_stack(x, 512, 3, stride=1, name='conv5')
    
    return x
