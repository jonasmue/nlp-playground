#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
"""

import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size=5, padding=1):
        """
        @param input_dim: the input dimension to the network, that is, the character embedding dimension
        @param output_dim: the output dimension, that is, number of filters. In the PDF this is denoted with f and set to the word embedding dimension
        @param kernel_size: the size of the kernel (window) sliding over the character sequence
        @param padding: padding size applied at each side of the the 1-d convolution
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.conv_layer = nn.Conv1d(input_dim, output_dim, kernel_size=kernel_size, padding=padding)

    def forward(self, input):
        """
        @param input: the input put through the CNN. In this case a batch of words of the shape (batch_size, character_embedding_size, m_word)
        @return x_conv_out: the convoluted and max-pooled batch of the shape (batch_size, word_embedding_size)
        """
        m_word = input.size(-1)
        x_conv = self.conv_layer(input)
        x_conv_out = torch.max_pool1d(torch.relu(x_conv),
                                      kernel_size=m_word - self.kernel_size + 1).squeeze(-1)
        return x_conv_out
