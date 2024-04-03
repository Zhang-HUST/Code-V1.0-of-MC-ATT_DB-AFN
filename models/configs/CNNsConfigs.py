num_classes = 7
Conv1 = {'in_channel': 1, 'filters': 32, 'kernel': (1, 5), 'stride': (1, 1), 'dilation': (1, 1), 'padding': 'same'}


def common_configs(modal):
    # sensors: 14 -- 补零-15 = 5*3
    if modal == 'E':
        pooling5_kernel, pooling6_kernel = (5, 1), (3, 1)
    # sensors: 15 -- 15 = 5*3
    elif modal == 'A':
        pooling5_kernel, pooling6_kernel = (5, 1), (3, 1)
    # sensors: 4 -- 4 = 2*2
    elif modal == 'G':
        pooling5_kernel, pooling6_kernel = (2, 1), (2, 1)
    # sensors: 18 -- 18 = 6*3
    elif modal == 'E-G':
        pooling5_kernel, pooling6_kernel = (6, 1), (3, 1)
    # sensors: 29 -- 补零-30 = 6*5
    elif modal == 'E-A':
        pooling5_kernel, pooling6_kernel = (6, 1), (5, 1)
    else:
        raise ValueError('modal error！')

    BN1 = {'use_BN': True}
    Activation1 = {'activation_type': 'relu'}
    Pooling1 = {'pooling_type': 'max2d', 'pooling_kernel': (1, 4), 'pooling_stride': (1, 4)}
    Dropout1 = {'use_dropout': True, 'drop_rate': 0.2}

    BN2 = {'use_BN': True}
    Activation2 = {'activation_type': 'relu'}
    Pooling2 = {'pooling_type': 'max2d', 'pooling_kernel': (1, 4), 'pooling_stride': (1, 4)}
    Dropout2 = {'use_dropout': True, 'drop_rate': 0.2}

    BN3 = {'use_BN': True}
    Activation3 = {'activation_type': 'relu'}
    Pooling3 = {'pooling_type': 'max2d', 'pooling_kernel': (1, 4), 'pooling_stride': (1, 4)}
    Dropout3 = {'use_dropout': True, 'drop_rate': 0.2}

    BN4 = {'use_BN': True}
    Activation4 = {'activation_type': 'relu'}
    Pooling4 = {'pooling_type': 'ave2d', 'pooling_kernel': (1, 4), 'pooling_stride': (1, 1)}
    Dropout4 = {'use_dropout': True, 'drop_rate': 0.2}

    BN5 = {'use_BN': True}
    Activation5 = {'activation_type': 'relu'}
    Pooling5 = {'pooling_type': 'max2d', 'pooling_kernel': pooling5_kernel, 'pooling_stride': pooling5_kernel}
    Dropout5 = {'use_dropout': True, 'drop_rate': 0.2}

    BN6 = {'use_BN': True}
    Activation6 = {'activation_type': 'relu'}
    Pooling6 = {'pooling_type': 'ave2d', 'pooling_kernel': pooling6_kernel, 'pooling_stride': (1, 1)}
    Dropout6 = {'use_dropout': True, 'drop_rate': 0.2}

    configs = (BN1, Activation1, Pooling1, Dropout1, BN2, Activation2, Pooling2, Dropout2, BN3, Activation3, Pooling3,
               Dropout3, BN4, Activation4, Pooling4, Dropout4, BN5, Activation5, Pooling5, Dropout5
               , BN6, Activation6, Pooling6, Dropout6)

    return configs


def cnn_configs():
    Conv2 = {'in_channel': 32, 'filters': 64, 'kernel': (1, 3), 'stride': (1, 1), 'dilation': (1, 1),
             'padding': 'same'}
    Conv3 = {'in_channel': 64, 'filters': 96, 'kernel': (1, 3), 'stride': (1, 1), 'dilation': (1, 1),
             'padding': 'same'}
    Conv4 = {'in_channel': 96, 'filters': 128, 'kernel': (1, 3), 'stride': (1, 1), 'dilation': (1, 1),
             'padding': 'same'}
    Conv5 = {'in_channel': 128, 'filters': 128, 'kernel': (3, 1), 'stride': (1, 1), 'dilation': (1, 1),
             'padding': 'same'}
    Conv6 = {'in_channel': 128, 'filters': 256, 'kernel': (3, 1), 'stride': (1, 1), 'dilation': (1, 1),
             'padding': 'same'}

    configs = (Conv1, Conv2, Conv3, Conv4, Conv5, Conv6)

    return configs

def linear_configs():
    linear1_in_dim = 256
    Linear1 = {'in_dim': linear1_in_dim, 'out_dim': 32}
    BN1_linear = {'use_BN': True}
    Activation1_linear = {'activation_type': 'relu'}
    Dropout1_linear = {'use_dropout': True, 'drop_rate': 0.2}

    Linear2 = {'in_dim': 32, 'out_dim': num_classes}
    BN2_linear = {'use_BN': False}
    Activation2_linear = {'activation_type': 'None'}
    Dropout2_linear = {'use_dropout': False, 'drop_rate': 0.2}

    configs = (Linear1, BN1_linear, Activation1_linear, Dropout1_linear, Linear2, BN2_linear, Activation2_linear,
               Dropout2_linear)

    return configs
