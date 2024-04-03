num_classes = 7


def common_configs():
    BN1 = {'use_BN': True}
    Activation1 = {'activation_type': 'relu'}
    Pooling1 = {'pooling_type': 'max2d', 'pooling_kernel': (1, 4), 'pooling_stride': (1, 4)}
    Dropout1 =  {'use_dropout': True, 'drop_rate': 0.2}

    BN2 = {'use_BN': True}
    Activation2 = {'activation_type': 'relu'}
    Pooling2 = {'pooling_type': 'max2d', 'pooling_kernel': (1, 4), 'pooling_stride': (1, 4)}
    Dropout2 =  {'use_dropout': True, 'drop_rate': 0.2}

    configs = (BN1, Activation1, Pooling1, Dropout1, BN2, Activation2, Pooling2, Dropout2)

    return configs


def cnn_configs():
    Conv1 = {'in_channel': 1, 'filters': 32, 'kernel': (1, 5), 'stride': (1, 1), 'dilation': (1, 1), 'padding': 'same'}
    Conv2 = {'in_channel': 32, 'filters': 64, 'kernel': (1, 3), 'stride': (1, 1), 'dilation': (1, 1),
             'padding': 'same'}
    configs = (Conv1, Conv2)

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
