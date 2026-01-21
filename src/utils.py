#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :utils.py
# @Time      :2025/9/4 16:54
# @Author    :ZHANG Yun in Rocket Force of University
# @Description: 
# @input     :
# @output    :
import torch
import pandas as pd
from Defaults import RNG
from obspy import UTCDateTime
from Defaults import MAX_LAT, MIN_LAT, MAX_LONG, MIN_LONG

class NormalizeTargets_LAT_LONG(torch.utils.data.Dataset):
    """
    Wraps a PyG dataset to normalize `y` from [min_val, max_val] → [-1, 1].
    把位置坐标归一化到-1到1之间
    """

    def __init__(self, dataset, min_lat=MIN_LAT, max_lat=MAX_LAT, min_long=MIN_LONG, max_long=MAX_LONG):
        self.dataset = dataset
        self.min_lat = min_lat
        self.max_lat = max_lat
        self.min_long = min_long
        self.max_long = max_long

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx].clone()
        # scale from [min, max] to [-1, 1]
        data.pos[:, 0] = 2 * (data.pos[:, 0] - self.min_lat) / (self.max_lat - self.min_lat) - 1
        data.pos[:, 1] = 2 * (data.pos[:, 1] - self.min_long) / (self.max_long - self.min_long) - 1
        data.event[:, 0] = 2 * (data.event[:, 0] - self.min_lat) / (self.max_lat - self.min_lat) - 1
        data.event[:, 1] = 2 * (data.event[:, 1] - self.min_long) / (self.max_long - self.min_long) - 1
        return data

    def denormalize(self, norm_y):
        """

        Args:
            norm_y: 经纬度为-1到1之间的值

        Returns:
            真实的值
        """
        norm_lat = norm_y[:, 0]
        norm_long = norm_y[:, 1]

        # 分别反归一化纬度和经度
        denorm_lat = 0.5 * (norm_lat + 1) * (self.max_lat - self.min_lat) + self.min_lat
        denorm_long = 0.5 * (norm_long + 1) * (self.max_long - self.min_long) + self.min_long

        # 合并结果
        denorm_y = torch.stack([denorm_lat, denorm_long], dim=1)
        return denorm_y


class NormalizeTargets_AZIMUTH_DISTANCE(torch.utils.data.Dataset):
    """
    Wraps a PyG dataset to normalize `y` from [min_val, max_val] → [-1, 1].
    把位置坐标归一化到-1到1之间
    """

    def __init__(self, dataset, min_azi, max_azi, min_distance, max_distance):
        self.dataset = dataset
        self.min_azi = min_azi
        self.max_azi = max_azi
        self.min_distance = min_distance
        self.max_distance = max_distance

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx].clone()
        # scale from [min, max] to [-1, 1]
        data.azimuth_distance[:, 0] = 2 * (data.azimuth_distance[:, 0] - self.min_azi) / (
                self.max_azi - self.min_azi) - 1
        data.azimuth_distance[:, 1] = 2 * (data.azimuth_distance[:, 1] - self.min_distance) / (
                self.max_distance - self.min_distance) - 1
        return data

    def denormalize(self, norm_y):
        """

        Args:
            norm_y: 经纬度为-1到1之间的值

        Returns:
            真实的值
        """
        norm_azi = norm_y[:, 0]
        norm_distance = norm_y[:, 1]

        # 分别反归一化纬度和经度
        denorm_azi = 0.5 * (norm_azi + 1) * (self.max_azi - self.min_azi) + self.min_azi
        denorm_distance = 0.5 * (norm_distance + 1) * (self.max_distance - self.min_distance) + self.min_distance

        # 合并结果
        denorm_y = torch.stack([denorm_azi, denorm_distance], dim=1)
        return denorm_y


def get_split_indices(dataset, train_end_time, valid_end_time):
    train_indices = []
    val_indices = []
    test_indices = []
    for i in range(len(dataset)):
        graph = dataset[i]
        pick_time = extract_pick_time(graph)  # 需要实现这个函数

        if pick_time < train_end_time:
            train_indices.append(i)
        elif train_end_time <= pick_time < valid_end_time:
            val_indices.append(i)
        else:
            test_indices.append(i)
    return train_indices, val_indices, test_indices


def extract_pick_time(graph):
    """提取pick时间，需要根据您的数据格式实现"""
    # 示例实现，请根据实际情况调整
    if hasattr(graph, 'pick') and graph.pick is not None:
        # 假设pick是Unix时间戳
        return UTCDateTime(graph.pick.mean().item())
