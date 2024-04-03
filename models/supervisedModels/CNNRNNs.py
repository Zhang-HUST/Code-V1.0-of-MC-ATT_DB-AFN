import torch.nn as nn
from models.commonBlocks.CNNLinearBlocks import CNNBlock, LinearBlock
from models.commonBlocks.SelfAttentions import MultiHeadAttention
from models.commonBlocks.ModalChannelAttention import ModalChannelAttention
from models.configs.RNNsConfigs import (common_configs, cnn_configs, linear_configs)
from trainTest.datasets.dataset_utils import get_basic_node_amount


class Conv2DRNNs(nn.Module):
    def __init__(self, modal, attention_type='MC-Att'):
        super(Conv2DRNNs, self).__init__()
        self.modal = modal
        self.attention_type = attention_type
        self.conv_type = ''.join(['CNN-BiLSTM-', self.attention_type])

        bascic_configs = common_configs()
        (BN1, Activation1, Pooling1, Dropout1, BN2, Activation2, Pooling2, Dropout2) = bascic_configs
        # CNNs
        configs = cnn_configs()
        (Conv1, Conv2) = configs
        self.cnn = nn.Sequential(
            CNNBlock(conv_dict=Conv1, bn_dict=BN1, activation_dict=Activation1, pooling_dict=Pooling1,
                     dropout_dict=Dropout1),
            CNNBlock(conv_dict=Conv2, bn_dict=BN2, activation_dict=Activation2, pooling_dict=Pooling2,
                     dropout_dict=Dropout2),
        )
        self.cnn_attention_layer = ModalChannelAttention(self.modal)

        # RNNs
        node_amount = get_basic_node_amount(modal)
        self.rnn = nn.LSTM(input_size=64*node_amount, hidden_size=128, num_layers=1, batch_first=True, bidirectional=True)
        self.rnn_attention_layer = MultiHeadAttention(d_model=256, num_heads=4)
        self.drop_out = nn.Dropout(p=0.2)

        fc_configs = linear_configs()
        (Linear1, BN1_linear, Activation1_linear, Dropout1_linear, Linear2, BN2_linear, Activation2_linear,
         Dropout2_linear) = fc_configs
        self.linear = nn.Sequential(
            LinearBlock(linear_dict=Linear1, bn_dict=BN1_linear, activation_dict=Activation1_linear,
                        dropout_dict=Dropout1_linear),
            LinearBlock(linear_dict=Linear2, bn_dict=BN2_linear, activation_dict=Activation2_linear,
                        dropout_dict=Dropout2_linear),
        )
        self.init_params()

    def forward(self, data):

        cnn_out = self.cnn(data)
        if self.attention_type == 'MC-Att':
            cnn_out = self.cnn_attention_layer(cnn_out)

        # 将CNN的输出形状调整为LSTM的输入形状
        # [32, 64, 29, 16] --  [32, 16, 29, 64] --  [32, 16, 64*29]
        cnn_out = cnn_out.permute(0, 3, 2, 1).contiguous().view(cnn_out.size(0), cnn_out.size(-1), -1)

        # RNN部分
        rnn_out, _ = self.rnn(cnn_out)
        # # 注意力机制
        query, key, value = rnn_out, rnn_out, rnn_out
        rnn_out, attn = self.rnn_attention_layer(query, key, value)
        rnn_out = self.drop_out(rnn_out)
        rnn_out = rnn_out[:, -1, :]
        # 输出层
        out = self.linear(rnn_out)
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




