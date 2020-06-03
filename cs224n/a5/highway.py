#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
"""

import torch
import torch.nn as nn


class Highway(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.proj_layer = nn.Linear(dim, dim)
        self.gate_layer = nn.Linear(dim, dim)

    def forward(self, input):
        x_proj = torch.relu(self.proj_layer(input))
        x_gate = torch.sigmoid(self.gate_layer(input))
        x_highway = x_gate * x_proj + (1 - x_gate) * input
        return x_highway
