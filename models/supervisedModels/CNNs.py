import torch.nn as nn
from models.commonBlocks.CNNLinearBlocks import CNNBlock, LinearBlock
from models.commonBlocks.ModalChannelAttention import ModalChannelAttention
from models.configs.CNNsConfigs import (common_configs, cnn_configs, linear_configs)


class GeneralCNNs(nn.Module):
    def __init__(self, modal, attention_type='MC-Att'):
        super(GeneralCNNs, self).__init__()
        self.modal = modal
        self.attention_type = attention_type
        self.conv_type = ''.join(['CNN-', self.attention_type])
        bascic_configs = common_configs(self.modal)
        (BN1, Activation1, Pooling1, Dropout1, BN2, Activation2, Pooling2, Dropout2, BN3, Activation3, Pooling3,
         Dropout3, BN4, Activation4, Pooling4, Dropout4, BN5, Activation5, Pooling5, Dropout5,
         BN6, Activation6, Pooling6, Dropout6) = bascic_configs

        fc_configs = linear_configs()
        (Linear1, BN1_linear, Activation1_linear, Dropout1_linear, Linear2, BN2_linear, Activation2_linear,
         Dropout2_linear) = fc_configs

        configs = cnn_configs()
        (Conv1, Conv2, Conv3, Conv4, Conv5, Conv6) = configs
        self.cnn_part1 = nn.Sequential(
            CNNBlock(conv_dict=Conv1, bn_dict=BN1, activation_dict=Activation1, pooling_dict=Pooling1,
                     dropout_dict=Dropout1),
            CNNBlock(conv_dict=Conv2, bn_dict=BN2, activation_dict=Activation2, pooling_dict=Pooling2,
                     dropout_dict=Dropout2),
        )
        self.cnn_part2 = nn.Sequential(
            CNNBlock(conv_dict=Conv3, bn_dict=BN3, activation_dict=Activation3, pooling_dict=Pooling3,
                     dropout_dict=Dropout3),
            CNNBlock(conv_dict=Conv4, bn_dict=BN4, activation_dict=Activation4, pooling_dict=Pooling4,
                     dropout_dict=Dropout4),
        )
        self.cnn_part3 = nn.Sequential(
            CNNBlock(conv_dict=Conv5, bn_dict=BN5, activation_dict=Activation5, pooling_dict=Pooling5,
                     dropout_dict=Dropout5),
            CNNBlock(conv_dict=Conv6, bn_dict=BN6, activation_dict=Activation6, pooling_dict=Pooling6,
                     dropout_dict=Dropout6),)

        self.attention_layer = ModalChannelAttention(self.modal)
        self.linear_part = nn.Sequential(
            LinearBlock(linear_dict=Linear1, bn_dict=BN1_linear, activation_dict=Activation1_linear,
                        dropout_dict=Dropout1_linear),
            LinearBlock(linear_dict=Linear2, bn_dict=BN2_linear, activation_dict=Activation2_linear,
                        dropout_dict=Dropout2_linear),
        )
        self.init_params()

    def forward(self, data):
        batch_size = data.shape[0]
        cnn_out_1 = self.cnn_part1(data)

        if self.attention_type == 'MC-Att':
            attention_out = self.attention_layer(cnn_out_1)
            cnn_out_2 = self.cnn_part2(attention_out)
        else:
            cnn_out_2 = self.cnn_part2(cnn_out_1)

        if self.modal in ['E', 'E-A']:
            cnn_out_2 = nn.ZeroPad2d((0, 0, 1, 0))(cnn_out_2)

        cnn_out_3 = self.cnn_part3(cnn_out_2)
        cnn_out_3 = cnn_out_3.reshape(batch_size, -1)

        out = self.linear_part(cnn_out_3)

        return out

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def get_model_name(self):
        return self.conv_type
