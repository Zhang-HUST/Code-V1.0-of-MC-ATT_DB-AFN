import os
import pandas as pd
import torch
import torch.nn as nn
from models.commonBlocks.CNNLinearBlocks import CNNBlock, LinearBlock
from models.commonBlocks.SelfAttentions import MultiHeadAttention
from models.commonBlocks.ModalChannelAttention import ModalChannelAttention
from models.configs.DBAFNConfigs import (common_cnn_configs, branch_cnn_configs, linear_configs)
from trainTest.datasets.dataset_utils import get_basic_node_amount


class DBAFNModels(nn.Module):
    def __init__(self, params):
        super(DBAFNModels, self).__init__()
        self.modal = params['modal']
        self.fuse_type = params['fuse_type']
        assert self.fuse_type == 'Add' or 'DyAdd'
        self.attention_type = params['attention_type']
        self.conv_type = ''.join(['CNN-BiLSTM-', self.fuse_type, '-', self.attention_type])

        configs = common_cnn_configs()
        (Conv1, BN1, Activation1, Pooling1, Dropout1, Conv2, BN2, Activation2, Pooling2, Dropout2) = configs
        # 通用的两层Conv
        self.common_cnn = nn.Sequential(
            CNNBlock(conv_dict=Conv1, bn_dict=BN1, activation_dict=Activation1, pooling_dict=Pooling1,
                     dropout_dict=Dropout1),
            CNNBlock(conv_dict=Conv2, bn_dict=BN2, activation_dict=Activation2, pooling_dict=Pooling2,
                     dropout_dict=Dropout2),
        )
        self.cnn_attention_layer = ModalChannelAttention(self.modal)

        # 分支CNN的进一步特征提取
        configs = branch_cnn_configs(self.modal)
        (Conv3, BN3, Activation3, Pooling3, Dropout3, Conv4, BN4, Activation4, Pooling4, Dropout4,
         Conv5, BN5, Activation5, Pooling5, Dropout5, Conv6, BN6, Activation6, Pooling6, Dropout6) = configs
        self.branch_cnn_1 = nn.Sequential(
            CNNBlock(conv_dict=Conv3, bn_dict=BN3, activation_dict=Activation3, pooling_dict=Pooling3,
                     dropout_dict=Dropout3),
            CNNBlock(conv_dict=Conv4, bn_dict=BN4, activation_dict=Activation4, pooling_dict=Pooling4,
                     dropout_dict=Dropout4), )
        self.branch_cnn_2 = nn.Sequential(
            CNNBlock(conv_dict=Conv5, bn_dict=BN5, activation_dict=Activation5, pooling_dict=Pooling5,
                     dropout_dict=Dropout5),
            CNNBlock(conv_dict=Conv6, bn_dict=BN6, activation_dict=Activation6, pooling_dict=Pooling6,
                     dropout_dict=Dropout6), )

        # 分支RNN的进一步特征提取
        node_amount = get_basic_node_amount(self.modal)
        self.rnn = nn.LSTM(input_size=64 * node_amount, hidden_size=128, num_layers=1, batch_first=True,
                           bidirectional=True)
        self.rnn_attention_layer = MultiHeadAttention(d_model=256, num_heads=4)
        self.rnn_drop_out = nn.Dropout(p=0.2)

        # 可训练的融合参数
        if self.fuse_type == 'DyAdd':
            self.fuse_weight = nn.Parameter(torch.randn(1, 2, requires_grad=True))
        # 分类器
        fc_configs = linear_configs()
        (Linear1, BN1_linear, Activation1_linear, Dropout1_linear, Linear2, BN2_linear, Activation2_linear,
         Dropout2_linear) = fc_configs
        self.fc_linear = nn.Sequential(
            LinearBlock(linear_dict=Linear1, bn_dict=BN1_linear, activation_dict=Activation1_linear,
                        dropout_dict=Dropout1_linear),
            LinearBlock(linear_dict=Linear2, bn_dict=BN2_linear, activation_dict=Activation2_linear,
                        dropout_dict=Dropout2_linear),
        )
        self.batch_size, self.device = None, None
        self.init_params()

    def forward(self, data):
        self.batch_size = data.shape[0]
        self.device = data.device
        # 通用的两层Conv
        common_cnn_out = self.common_cnn(data)
        if self.attention_type == 'MC-Att':
            common_cnn_out = self.cnn_attention_layer(common_cnn_out)

        multi_branch_outs = []
        # CNN
        branch_out_1 = self.branch_cnn_1(common_cnn_out)
        if self.modal in ['E', 'E-A']:
            branch_out_1 = nn.ZeroPad2d((0, 0, 1, 0))(branch_out_1)
        branch_out_2 = self.branch_cnn_2(branch_out_1)
        # [32, 256]
        branch_out = branch_out_2.reshape(self.batch_size, -1)
        multi_branch_outs.append(branch_out)

        # RNN
        cnn_out = common_cnn_out.permute(0, 3, 2, 1).contiguous().view(common_cnn_out.size(0),common_cnn_out.size(-1), -1)
        rnn_out, _ = self.rnn(cnn_out)
        # 注意力机制
        query, key, value = rnn_out, rnn_out, rnn_out
        rnn_out, attn = self.rnn_attention_layer(query, key, value)
        rnn_out = self.rnn_drop_out(rnn_out)
        branch_out = rnn_out[:, -1, :]
        multi_branch_outs.append(branch_out)

        if self.fuse_type == 'Add':
            stacked_tensor = torch.stack(multi_branch_outs, dim=-1)
            summed_tensor = torch.sum(stacked_tensor, dim=-1)
        else:
            # learn_AWMF_weight = nn.Sigmoid()(self.fuse_weight).to(device=self.device)
            learn_fuse_weight = nn.Softmax(dim=1)(self.fuse_weight).to(device='cpu')
            summed_tensor = torch.zeros(self.batch_size, branch_out.shape[-1]).to(device=self.device)
            for i in range(2):
                summed_tensor = multi_branch_outs[i] * learn_fuse_weight[0, i] + summed_tensor
        summed_tensor.to(device=self.device)
        out = self.fc_linear(summed_tensor)
        return out

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LSTM) or isinstance(m, nn.GRU):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_normal_(param)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0)

    def get_model_name(self):
        return self.conv_type

