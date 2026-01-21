#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :GraphNet_Sta_Event.py
# @Time      :2025/9/3 8:06
# @Author    :ZHANG Yun in Rocket Force of University
# @Description: 
# @input     :
# @output    :
from torch_geometric.nn import GCNConv, MessagePassing, MLP, GATv2Conv, global_mean_pool
from torch import nn
import torch
import torch.nn.functional as F
from torch_geometric.nn.aggr import AttentionalAggregation
import numpy as np

# 伪代码示例 (使用PyTorch风格)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import softmax as pygsoftmax


class ConvBlock(nn.Module):
    def __init__(self, in_channels=3, kernel_size=2, stride=1, padding="same", dropout=0.2, pooling_size=2,
                 output_size=64):
        super(ConvBlock, self).__init__()
        self.out_channels1 = 18
        self.conv1 = nn.Conv2d(in_channels, self.out_channels1, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm2d(self.out_channels1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=pooling_size)
        self.dropout = nn.Dropout2d(dropout)

        self.out_channels2 = 36
        self.conv2 = nn.Conv2d(self.out_channels1, self.out_channels2, kernel_size, stride, padding)
        self.bn2 = nn.BatchNorm2d(self.out_channels2)

        self.out_channels3 = 54
        self.conv3 = nn.Conv2d(self.out_channels2, self.out_channels3, kernel_size, stride, padding)
        self.bn3 = nn.BatchNorm2d(self.out_channels3)

        self.out_channels4 = 54
        self.conv4 = nn.Conv2d(self.out_channels3, self.out_channels4, kernel_size, stride, padding)
        self.bn4 = nn.BatchNorm2d(self.out_channels4)

        self.fc = nn.Linear(54 * 3 * 2, out_features=output_size)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.dropout(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.dropout(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.dropout(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.dropout(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class AttentionalAggregationWithWeights(AttentionalAggregation):
    def __init__(self, gate_nn):
        super().__init__(gate_nn=gate_nn)

    def forward(self, x, index=None, **kwargs):
        if index is None:
            index = x.new_zeros(x.size(0), dtype=torch.long)

        # 计算注意力权重
        g = self.gate_nn(x).view(-1)

        attention_weights = pygsoftmax(g, index)

        # 使用父类方法计算池化结果（更简洁可靠）
        pooled = super().forward(x, index, **kwargs)

        return pooled, attention_weights


class GraphNet(torch.nn.Module):
    """
    GNN model with MLP → 2x GATv2Conv → MLP for graph-level origin prediction.

    Args:
        channels_x (int): Positional input features per node.
        channels_y (int): Signal features per node.
        hidden_channels (int): Hidden dimension.
        dropout (float): Dropout probability.
        self_loops (bool): Whether to add self-loops to GATv2Conv.

    Output:
        Tensor of shape (batch_size, 2), normalized origin coordinates (in [-1, 1]).
    """

    def __init__(self, in_channels=3, conv_out_features=64, dropout=0.2,
                 self_loops=True):
        super(GraphNet, self).__init__()
        torch.manual_seed(1234)

        self.conv_spec = ConvBlock(in_channels=in_channels, kernel_size=3, stride=1, padding='same', dropout=dropout,
                                   pooling_size=2, output_size=conv_out_features)

        self.conv_graph = GATv2Conv(
            in_channels=conv_out_features,
            out_channels=conv_out_features,
            heads=2,  # 多头注意力机制
            edge_dim=1,  # edge_weight(node,1) 会同时考虑相似度与距离
            add_self_loops=self_loops,  # 算新的特征是否考虑自己的特征
            concat=True  # 多头注意力机制拼接
        )

        self.norm1 = nn.LayerNorm(conv_out_features * 2)  # 匹配conv1输出

        self.dropout = nn.Dropout(dropout)

        self.att_pool = AttentionalAggregationWithWeights(
            gate_nn=torch.nn.Sequential(
                nn.Linear(conv_out_features * 2, 1),
                nn.ReLU()
            )
        )

        self.out_linear = nn.Linear(conv_out_features * 2, 2)

    def forward(self, event, pos, signal, azi_distance, edge_index, edge_weight, batch=None):
        x = signal
        x = self.conv_spec(x)
        x = self.conv_graph(x, edge_index, edge_weight)
        x = F.relu(self.norm1(x))
        x = self.dropout(x)
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        graph_repr, weights = self.att_pool(x, batch)
        return self.out_linear(graph_repr)


class EarlyStopper:
    """
    A class for early stopping the training process when the validation loss stops improving.
    """

    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
