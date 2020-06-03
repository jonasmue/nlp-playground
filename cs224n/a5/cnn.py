#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
"""

import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, input_dim, output_dim, m_word, kernel_size=5, padding=1):
        super().__init__()
        self.m_word = m_word
        self.kernel_size = kernel_size
        self.conv_layer = nn.Conv1d(input_dim, output_dim, kernel_size=kernel_size, padding=padding)

    def forward(self, input):
        x_conv = self.conv_layer(input)
        x_conv_out = torch.max_pool1d(torch.relu(x_conv),
                                      kernel_size=self.m_word - self.kernel_size + 1).squeeze(-1)
        return x_conv_out
