from torch import nn
import torch
from torch.nn import functional as F
import numpy as np

from src.model2.Residual import ResidualStack


class Decoder(nn.Module):
    def __init__(self, in_channels=1, num_hiddens=128, num_residual_layers=2, num_residual_hiddens=32):
        super().__init__()
        self._unpool = nn.ConvTranspose2d(in_channels=1,
                                                out_channels=num_hiddens,
                                                kernel_size=3,
                                                stride=1, padding=1)
        self._residual_stack2 = ResidualStack(in_channels=num_hiddens,
                                             num_hiddens=num_hiddens,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hiddens=num_residual_hiddens)
        self._conv_trans_1 = nn.ConvTranspose2d(in_channels=num_hiddens,
                                                out_channels=num_hiddens//2,
                                                kernel_size=4,
                                                stride=2, padding=1)
        self._conv_trans_2 = nn.ConvTranspose2d(in_channels=num_hiddens//2,
                                                out_channels=in_channels,
                                                kernel_size=4,
                                                stride=2, padding=1)
        self._activation = nn.LeakyReLU()
        self.final_act = nn.Tanh()

    def forward(self, x):
        x = x.view(-1, 1, 32, 32)
        x = self._unpool(x)
        x = self._residual_stack2(x)
        x = self._activation(x)
        x = self._conv_trans_1(x)
        x = self._activation(x)
        x = self._conv_trans_2(x)
        x = self.final_act(x)
        return x