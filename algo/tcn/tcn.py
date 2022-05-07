"""
Implementation of "Temporal Convolutional Network (TCN)"
https://hongl.tistory.com/253
"""

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import pdb


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class ResidualBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2, activation='relu'):
        super(ResidualBlock, self).__init__()
        self.activation_map = {"relu": nn.ReLU(), "tanh": nn.Tanh(), "leakyrelu": nn.LeakyReLU()}
        assert activation in list(self.activation_map.keys()), "Unavailable activation."

        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.activation1 = self.activation_map[activation]
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.activation2 = self.activation_map[activation]
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.activation1, self.dropout1,
            self.conv2, self.chomp2, self.activation2, self.dropout2
        )

        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.activation = self.activation_map[activation]
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.activation(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, stride=1, dropout=0.2, activation='relu'):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers.append(ResidualBlock(in_channels, out_channels, kernel_size, stride=stride, dilation=dilation_size,
                                        padding=(kernel_size - 1) * dilation_size, dropout=dropout, activation=activation))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
