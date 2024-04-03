import math
import torch
import torch.nn as nn
from trainTest.datasets.dataset_utils import get_basic_node_amount
import torch.nn.functional as F


class MultiModalAttention(nn.Module):
    def __init__(self, modal):
        super(MultiModalAttention, self).__init__()
        assert modal == 'E-A'
        self.node_modal_E = get_basic_node_amount(modal='E')
        self.node_modal = get_basic_node_amount(modal=modal)
        self.conv = nn.Conv1d(self.node_modal, 2, kernel_size=1)
        self.channel_att_layer = ChannelAttention(channel=self.node_modal, gamma=2, b=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        permute_x = x.permute(0, 2, 1, 3).contiguous()

        # 分支1
        x1 = torch.mean(permute_x, dim=(2, 3))
        x1 = self.conv(x1.unsqueeze(2)).squeeze(2)
        x1 = self.sigmoid(x1)
        repeated_first_neuron = x1[:, 0].unsqueeze(1).repeat(1, self.node_modal_E)
        repeated_second_neuron = x1[:, 1].unsqueeze(1).repeat(1, h-self.node_modal_E)
        x1 = torch.cat((repeated_first_neuron, repeated_second_neuron), dim=1).unsqueeze(-1).unsqueeze(-1)

        # 分支2
        x2 = self.channel_att_layer(permute_x)

        x3 = torch.add(x1, x2)
        out = permute_x * x3.expand_as(permute_x)
        out = out.permute(0, 2, 1, 3).contiguous()

        return out


class UniModalAttention(nn.Module):
    def __init__(self, modal):
        super(UniModalAttention, self).__init__()
        assert modal in ['E', 'A']
        self.node_modal = get_basic_node_amount(modal=modal)
        self.channel_att_layer = ChannelAttention(channel=self.node_modal, gamma=2, b=1)

    def forward(self, x):
        permute_x = x.permute(0, 2, 1, 3).contiguous()

        # 分支2
        x2 = self.channel_att_layer(permute_x)
        out = permute_x*x2.expand_as(permute_x)
        out = out.permute(0, 2, 1, 3).contiguous()

        return out


class ChannelAttention(nn.Module):
    def __init__(self, channel, gamma=2, b=1):
        super(ChannelAttention, self).__init__()
        t = int(abs((math.log(channel, 2) + b) / gamma))
        k = t if t % 2 else t + 1
        k_size = max(3, k)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding='same', bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)
        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        # Multi-scale information fusion
        y = self.sigmoid(y)

        return y


class ModalChannelAttention(nn.Module):
    def __init__(self, modal):
        super(ModalChannelAttention, self).__init__()
        assert modal in ['E', 'A', 'E-A']
        self.modal = modal
        if self.modal in ['E', 'A']:
            self.modalAttLayer = UniModalAttention(modal=self.modal)
        else:
            self.modalAttLayer = MultiModalAttention(modal=self.modal)
        self.channel_att_layer = ChannelAttention(channel=64, gamma=2, b=1)

    def forward(self, x):
        channelAtt = self.channel_att_layer(x)
        channelAttFeatures = x*channelAtt.expand_as(x)

        modalAttFeatures = self.modalAttLayer(x)
        fuseFeatures = channelAttFeatures + modalAttFeatures

        return fuseFeatures


