#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :Graph_Datasets.py
# @Time      :2025/9/3 8:03
# @Author    :ZHANG Yun in Rocket Force of University
# @Description: 
# @input     :
# @output    :
import os.path
import random

from obspy import UTCDateTime
import matplotlib.pyplot as plt
from geopy.distance import geodesic
import copy
import numpy as np

import Defaults
from Defaults import (WINDOWS, SAMPLING_RATE, label_mapping, NPERSEG, NOVERLAP_SAMPLES,
                      RAND_SEED, epsilon, FREQMAX, FREQMIN)
import multiprocessing as mp
from functools import partial
import torch
from torch_geometric.data import Data, InMemoryDataset
from geopy.point import Point
from scipy import signal
from obspy.geodetics import gps2dist_azimuth
import obspy


def check(tr):
    if len(tr.data) > WINDOWS * SAMPLING_RATE:
        tr = tr.slice(starttime=tr.stats.starttime,
                      endtime=tr.stats.endtime - 1.0 / SAMPLING_RATE)
    elif len(tr.data) < WINDOWS * SAMPLING_RATE:
        tr = tr.slice(starttime=tr.stats.starttime,
                      endtime=tr.stats.endtime + 1.0 / SAMPLING_RATE)
    return tr


def add_azimuth_distance(lat1, lon1, lat2, lon2):
    point1 = Point(latitude=lat1, longitude=lon1)
    point2 = Point(latitude=lat1, longitude=lon2)
    distance_km = geodesic(point1, point2).km

    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)

    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)

    dlon = lon2_rad - lon1_rad
    y = np.sin(dlon) * np.cos(lat2_rad)
    x = np.cos(lat1_rad) * np.sin(lat2_rad) - np.sin(lat1_rad) * np.cos(lat2_rad) * np.cos(dlon)
    azimuth_rad = np.arctan2(y, x)
    azimuth_deg = np.degrees(azimuth_rad)

    azimuth = (azimuth_deg + 360) % 360
    return [azimuth, distance_km]


def add_edge_weight(g):
    """
    Compute edge weights based on inverse distance between nodes.

    This function adds an 'edge_weight' attribute to each graph,
    calculated as 1 / (distance + 1) to ensure numerical stability.

    Args:
        g (Data): PyG graph with 'pos' and 'edge_index'

    Returns:
        Data: Modified graph with 'edge_weight'
    """
    edge_weight = []
    for edge in g.edge_index.T:
        node_a, node_b = g.pos[edge[0]], g.pos[edge[1]]
        dist = geodesic((node_a[0], node_a[1]), (node_b[0], node_b[1])).kilometers
        dist = np.maximum(dist, 1e-6)
        edge_weight.append(1 / dist)  # add 1 to avoid division by zero
    g.edge_weight = torch.tensor(np.array(edge_weight)).type(torch.FloatTensor)
    return g


class GraphDataset(InMemoryDataset):
    """
    Constructing dataset for training GNNs to classification.

    Each graph in the dataset includes:
    - Randomly placed nodes in 2D space.
    - waveform signals at each node, delayed by distance from a hidden origin.

    Args:
        root (str): Root directory to save the processed data
        nb_graph (int): Number of synthetic graphs to generate
    """

    def __init__(self, root, transform=None, pre_transform=None, csv=None, augument=False):
        self.csv = csv
        self.event = self.csv["id"].unique()
        self.nb_graph = len(self.event)
        self.augument = augument
        super(GraphDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_file_names(self):
        # No raw input files; data is generated from scratch
        return 0

    @property
    def processed_file_names(self):
        return 'data.pt'

    def process(self):
        data_list = []

        for i, ev in enumerate(self.event):
            if i % 100 == 0:
                print(f"正在生成第{i}个图")
            ev_csv = self.csv[self.csv['id'] == ev].reset_index(drop=True)
            nb_nodes = len(ev_csv)
            evla = ev_csv['evla'].iloc[0]
            evlo = ev_csv['evlo'].iloc[0]
            evdp = ev_csv['eldp'].iloc[0]
            evid = ev_csv['id'].iloc[0]
            la = label_mapping[ev_csv['type'].iloc[0]]
            origin_tensor = torch.tensor([evla, evlo, evdp], dtype=torch.float)
            evid_tensor = torch.tensor([evid], dtype=torch.int64)
            label = torch.tensor([la], dtype=torch.int64)
            spec_list, stla, stlo, distance_in_km_list, azimuth_list, picktime, nb_nodes = self.Station_infos(nb_nodes,
                                                                                                              ev_csv)
            if nb_nodes == 0:
                continue
            azimuth_distance = torch.tensor([[x, y] for x, y in zip(azimuth_list, distance_in_km_list)],
                                            dtype=torch.float)
            signal = torch.tensor(np.array(spec_list), dtype=torch.float).reshape(nb_nodes, 3, 48, 40)

            pick = torch.tensor(np.array(picktime), dtype=torch.float).reshape(nb_nodes, 1)

            pos = torch.tensor([[x, y] for x, y in zip(stla, stlo)], dtype=torch.float)

            g = Data(pos=pos, signal=signal, y=label, event=origin_tensor.unsqueeze(0),
                     azimuth_distance=azimuth_distance, evid=evid_tensor.unsqueeze(0), pick=pick)
            data_list.append(g)

        # Apply graph transformations if provided (e.g., KNN, edge weights)
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
            data_list = [add_edge_weight(data) for data in data_list]

        # Save to disk
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def Station_infos(self, nb_nodes, ev_csv):
        def Get_trace(csv, j):
            id, net, sta, chan = int(csv["id"].iloc[j]), csv['net'].iloc[j], csv['sta'].iloc[j], csv['chan'].iloc[j]
            evla, evlo, stla, stlo = csv["evla"].iloc[j], csv["evlo"].iloc[j], csv["stla"].iloc[j], csv["stlo"].iloc[j]
            # 注意：gps2dist_azimuth 返回的是 (distance_in_meters, azimuth, back_azimuth)
            distance_in_meters, azimuth, _ = gps2dist_azimuth(evla, evlo, stla, stlo)
            filenameZ = os.path.join(Defaults.SAC_ROOT, f"{id}.0.{net}.{sta}.{chan}.SAC")
            filenameE = os.path.join(Defaults.SAC_ROOT, f"{id}.0.{net}.{sta}.{chan[:2]}E.SAC")
            filenameN = os.path.join(Defaults.SAC_ROOT, f"{id}.0.{net}.{sta}.{chan[:2]}N.SAC")
            filenames = [filenameZ, filenameE, filenameN]
            try:
                st = obspy.Stream()
                for file in filenames:
                    st += obspy.read(file)
                st = st.resample(SAMPLING_RATE)
                st = st.copy().rotate(method='NE->RT', back_azimuth=azimuth)
                return st, distance_in_meters, azimuth
            except:
                try:
                    filenameZ = os.path.join(Defaults.SAC_ROOT, f"{id}.0.{net}.{sta}.{chan}.SAC")
                    filenameE = os.path.join(Defaults.SAC_ROOT, f"{id}.0.{net}.{sta}.{chan[:2]}2.SAC")
                    filenameN = os.path.join(Defaults.SAC_ROOT, f"{id}.0.{net}.{sta}.{chan[:2]}1.SAC")
                    filenames = [filenameZ, filenameE, filenameN]
                    st = obspy.Stream()
                    for file in filenames:
                        st += obspy.read(file)
                    st = st.resample(SAMPLING_RATE)
                    st = st.copy().rotate(method='NE->RT', back_azimuth=azimuth)
                    return st, distance_in_meters, azimuth
                except:
                    st = []
                    st.append(obspy.read(filenameZ)[0])
                    st[0] = st[0].resample(SAMPLING_RATE)
                    st.append(None)
                    st.append(None)
                    return st, distance_in_meters, azimuth

        def stft_spectrogram(tr):
            if tr:
                tr = tr.detrend('constant').detrend('linear').taper(0.01).filter('highpass', freq=1)
                f, t, spec = signal.stft(tr.data, tr.stats.sampling_rate, nperseg=NPERSEG,
                                         noverlap=NOVERLAP_SAMPLES, window='hann')
                freq_mask = (f >= FREQMIN) & (f <= FREQMAX)
                f_selected = f[freq_mask]
                t_mask = (t > 0) & (t < WINDOWS)
                t_selected = t[t_mask]
                spec_freq_selected = spec[freq_mask, :]
                spec_freq_t_selected = spec_freq_selected[:, t_mask]
                magnitude_spectrogram = np.abs(spec_freq_t_selected)
                # 对数缩放 (转换为分贝刻度)
                log_spectrogram = 10 * np.log10(magnitude_spectrogram + epsilon)
                # 使用最大值归一化
                min_val = np.min(log_spectrogram)
                max_val = np.max(log_spectrogram)
                if np.isclose(max_val, min_val):
                    normalized_spectrogram = np.zeros_like(log_spectrogram)

                else:
                    normalized_spectrogram = (log_spectrogram - min_val) / (max_val - min_val)
                return normalized_spectrogram
            else:
                return np.zeros((48, 40))

        stla = []
        stlo = []
        picktime = []
        spec_list = []
        distance_in_km_list = []
        azimuth_list = []
        N = nb_nodes
        for j in range(nb_nodes):
            spec_list_sta = []
            st, distance_in_meters, azimuth = Get_trace(ev_csv, j)
            if st[0].stats.endtime - st[0].stats.starttime < WINDOWS:
                N = N - 1
                continue
            for tr in st:
                spec_list_sta.append(stft_spectrogram(tr))
            spec_array_sta = np.array(spec_list_sta)
            distance_in_km_list.append(distance_in_meters / 1000.0)
            azimuth_list.append(azimuth)
            stla.append(ev_csv['stla'].iloc[j])
            stlo.append(ev_csv['stlo'].iloc[j])
            picktime.append(UTCDateTime(ev_csv['picktime'].iloc[j]).timestamp)
            spec_list.append(spec_array_sta)
        return spec_list, stla, stlo, distance_in_km_list, azimuth_list, picktime, N


def visualize_graph_torch(g, color, use_method='energy', pred=False, ax=None, title=None):
    """
    Visualize the graph structure with PyTorch Geometric graph object.

    Args:
        g (Data): Graph data object with edge_index, pos, and feature color
        color (str): Node attribute key to use for coloring
        pred (bool): If True, also plot predicted origin with blue cross
        ax (matplotlib.axes.Axes, optional): If provided, draw into this axis
        title (str, optional): Optional title to display above plot

    Behavior:
    - Nodes are colored by the specified feature (e.g., signal or attention score)
    - Edges are drawn in blue
    - True origin marked with red ❌
    - Prediction (if enabled) marked with blue ❌
    """

    if use_method == 'first':
        color_values = g[color][:, 0]
    elif use_method == 'mean':
        color_values = g[color].abs().mean(dim=1)
    elif use_method == 'max':
        color_values = g[color].abs().max(dim=1).values
    elif use_method == 'energy':
        color_values = (g[color] ** 2).sum(dim=1)
    else:
        color_values = g[color][:, 0]
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
        created_fig = True

    # Plot edges
    for edge in g.edge_index.T:
        ax.plot(
            [g.pos[edge[0]][0], g.pos[edge[1]][0]],
            [g.pos[edge[0]][1], g.pos[edge[1]][1]],
            color='blue', linewidth=1
        )

    # Node scatter with color
    scatter = ax.scatter(
        x=g.pos.T[0],
        y=g.pos.T[1],
        alpha=1,
        c=color_values,
        s=150
    )

    # True origin
    if hasattr(g, 'y') and g.y[0].numel() == 2:
        ax.plot(g.y[0][0], g.y[0][1], 'rx', markersize=12, markeredgewidth=3)

    # Predicted origin
    if pred and hasattr(g, 'pred') and g.pred[0].numel() == 2:
        ax.plot(g.pred[0][0], g.pred[0][1], 'bx', markersize=12, markeredgewidth=3)

    # Title if given
    if title:
        ax.set_title(title)

    # Clean axes
    ax.set_xticks([])
    ax.set_yticks([])

    # Legend for coloring only once if top-level figure
    if created_fig:
        legend1 = ax.legend(*scatter.legend_elements(), loc='center left', bbox_to_anchor=(1, 0.5))
        ax.add_artist(legend1)
        plt.show()


def process_data(data, pre_transform_func, add_edge_weight_func):
    """处理单个数据的函数"""
    data = pre_transform_func(data)
    data = add_edge_weight_func(data)
    return data


class GraphDataset_Multi(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, csv=None, augument=False):
        self.csv = csv
        self.event = self.csv["id"].unique()
        self.nb_graph = len(self.event)
        print(f"总共有 {self.nb_graph} 个事件需要处理")
        self.augument = augument
        super(GraphDataset_Multi, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_file_names(self):
        # No raw input files; data is generated from scratch
        return 0

    @property
    def processed_file_names(self):
        return 'data.pt'

    def process(self):
        # 创建进程池
        # num_workers = mp.cpu_count() - 2  # 使用所有CPU核心
        print(f"使用 {Defaults.NUM_WORKS} 个进程进行并行处理")

        # 准备参数
        process_func = partial(self.process_single_event, csv=self.csv)

        # 使用进程池并行处理
        with mp.Pool(processes=Defaults.NUM_WORKS) as pool:
            results = pool.map(process_func, enumerate(self.event))

        data_list = []
        for result in results:
            if result is not None:
                if isinstance(result, list):
                    for re in result:
                        if re is not None:
                            data_list.append(re)
                else:
                    data_list.append(result)

        # 使用多进程处理
        if self.pre_transform is not None:
            with mp.Pool(processes=Defaults.NUM_WORKS) as pool:
                # 使用partial固定函数参数
                process_func = partial(process_data,
                                       pre_transform_func=self.pre_transform,
                                       add_edge_weight_func=add_edge_weight)
                data_list = pool.map(process_func, data_list)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def process_single_event(self, args, csv):
        """处理单个事件的函数（用于多进程）"""

        def produce_graph(n_sample, df_sample):
            spec_list, stla, stlo, distance_in_km_list, azimuth_list, picktime, nb_nodes = self.Station_infos(n_sample,
                                                                                                              df_sample)

            if nb_nodes == 0:
                print(f"事件 {df_sample["id"].iloc[0]} 没有有效台站数据，跳过。")
                return None

            azimuth_distance = torch.tensor([[x, y] for x, y in zip(azimuth_list, distance_in_km_list)],
                                            dtype=torch.float)
            signal = torch.tensor(np.array(spec_list), dtype=torch.float).reshape(nb_nodes, 3, 48, 40)
            pick = torch.tensor(np.array(picktime), dtype=torch.float).reshape(nb_nodes, 1)
            pos = torch.tensor([[x, y] for x, y in zip(stla, stlo)], dtype=torch.float)

            g = Data(
                pos=pos, signal=signal, y=label, event=origin_tensor.unsqueeze(0),
                azimuth_distance=azimuth_distance, evid=evid_tensor.unsqueeze(0), pick=pick
            )

            return g

        i, ev = args
        if i % 100 == 0:
            print(f"正在处理第{i}个事件")

        try:
            ev_csv = csv[csv['id'] == ev].reset_index(drop=True)
            nb_nodes = len(ev_csv)

            if nb_nodes == 0:
                print(f"事件 {ev} 没有有效台站数据，跳过。")

                return None

            evla = ev_csv['evla'].iloc[0]
            evlo = ev_csv['evlo'].iloc[0]
            evdp = ev_csv['Depth'].iloc[0]
            evid = ev_csv['id'].iloc[0]
            la = label_mapping[ev_csv['Typ'].iloc[0]]

            origin_tensor = torch.tensor([evla, evlo, evdp], dtype=torch.float)
            evid_tensor = torch.tensor([evid], dtype=torch.int64)
            label = torch.tensor([la], dtype=torch.int64)
            random.seed(42)
            if la == 0 and nb_nodes > 5 and UTCDateTime(
                    ev_csv["picktime"].iloc[0]) < Defaults.Train_End_Time and self.augument:
                graphs = []
                for augu_index in range(Defaults.EQ_Augument_Samples):
                    nb_sample = random.randint(1, nb_nodes)
                    df_sample = ev_csv.sample(n=nb_sample, random_state=RAND_SEED + augu_index).reset_index(drop=True)
                    graphs.append(produce_graph(nb_sample, df_sample))
                return graphs
            elif la == 1 and nb_nodes > 5 and UTCDateTime(
                    ev_csv["picktime"].iloc[0]) < Defaults.Train_End_Time and self.augument:
                graphs = []
                for augu_index in range(Defaults.QB_Augument_Samples):
                    nb_sample = random.randint(1, nb_nodes)
                    df_sample = ev_csv.sample(n=nb_sample, random_state=RAND_SEED + augu_index).reset_index(drop=True)
                    graphs.append(produce_graph(nb_sample, df_sample))
                return graphs
            else:
                return (produce_graph(nb_nodes, ev_csv))
        except Exception as e:
            print(f"处理事件 {ev} 时出错: {e}")
            return None

    def Station_infos(self, nb_nodes, ev_csv):
        def Get_trace(csv, j):
            id, net, sta, chan = int(csv["id"].iloc[j]), csv['net'].iloc[j], csv['sta'].iloc[j], csv['chan'].iloc[j]
            evla, evlo, stla, stlo = csv["evla"].iloc[j], csv["evlo"].iloc[j], csv["stla"].iloc[j], csv["stlo"].iloc[j]
            # 注意：gps2dist_azimuth 返回的是 (distance_in_meters, azimuth, back_azimuth)
            distance_in_meters, azimuth, _ = gps2dist_azimuth(evla, evlo, stla, stlo)
            filenameZ = os.path.join(Defaults.SAC_ROOT, f"{id}.0.{net}.{sta}.{chan}.SAC")
            filenameE = os.path.join(Defaults.SAC_ROOT, f"{id}.0.{net}.{sta}.{chan[:2]}E.SAC")
            filenameN = os.path.join(Defaults.SAC_ROOT, f"{id}.0.{net}.{sta}.{chan[:2]}N.SAC")
            filenames = [filenameZ, filenameE, filenameN]
            try:
                st = obspy.Stream()
                for file in filenames:
                    st += obspy.read(file)
                st = st.resample(SAMPLING_RATE)
                st = st.copy().rotate(method='NE->RT', back_azimuth=azimuth)
                return st, distance_in_meters, azimuth
            except:
                try:
                    filenameZ = os.path.join(Defaults.SAC_ROOT, f"{id}.0.{net}.{sta}.{chan}.SAC")
                    filenameE = os.path.join(Defaults.SAC_ROOT, f"{id}.0.{net}.{sta}.{chan[:2]}2.SAC")
                    filenameN = os.path.join(Defaults.SAC_ROOT, f"{id}.0.{net}.{sta}.{chan[:2]}1.SAC")
                    filenames = [filenameZ, filenameE, filenameN]
                    st = obspy.Stream()
                    for file in filenames:
                        st += obspy.read(file)
                    st = st.resample(SAMPLING_RATE)
                    st = st.copy().rotate(method='NE->RT', back_azimuth=azimuth)
                    return st, distance_in_meters, azimuth
                except:
                    filenameZ = os.path.join(Defaults.SAC_ROOT, f"{id}.{net}.{sta}.{chan}.SAC")
                    filenameE = os.path.join(Defaults.SAC_ROOT, f"{id}.{net}.{sta}.{chan[:2]}E.SAC")
                    filenameN = os.path.join(Defaults.SAC_ROOT, f"{id}.{net}.{sta}.{chan[:2]}N.SAC")
                    filenames = [filenameZ, filenameE, filenameN]
                    try:
                        st = obspy.Stream()
                        for file in filenames:
                            st += obspy.read(file)
                        st = st.resample(SAMPLING_RATE)
                        st = st.copy().rotate(method='NE->RT', back_azimuth=azimuth)
                        return st, distance_in_meters, azimuth
                    except:
                        try:
                            filenameZ = os.path.join(Defaults.SAC_ROOT, f"{id}.{net}.{sta}.{chan}.SAC")
                            filenameE = os.path.join(Defaults.SAC_ROOT, f"{id}.{net}.{sta}.{chan[:2]}2.SAC")
                            filenameN = os.path.join(Defaults.SAC_ROOT, f"{id}.{net}.{sta}.{chan[:2]}1.SAC")
                            filenames = [filenameZ, filenameE, filenameN]
                            st = obspy.Stream()
                            for file in filenames:
                                st += obspy.read(file)
                            st = st.resample(SAMPLING_RATE)
                            st = st.copy().rotate(method='NE->RT', back_azimuth=azimuth)
                            return st, distance_in_meters, azimuth
                        except:
                            try:
                                st = []
                                filenameZ = os.path.join(Defaults.SAC_ROOT, f"{id}.0.{net}.{sta}.{chan}.SAC")
                                st.append(obspy.read(filenameZ)[0])
                                st[0] = st[0].resample(SAMPLING_RATE)
                                st.append(None)
                                st.append(None)
                                return st, distance_in_meters, azimuth
                            except:
                                try:
                                    st = []
                                    filenameZ = os.path.join(Defaults.SAC_ROOT, f"{id}.{net}.{sta}.{chan}.SAC")
                                    st.append(obspy.read(filenameZ)[0])
                                    st[0] = st[0].resample(SAMPLING_RATE)
                                    st.append(None)
                                    st.append(None)
                                    return st, distance_in_meters, azimuth
                                except:
                                    return None, None, None

        def stft_spectrogram(tr):
            if tr:
                tr = tr.detrend('constant').detrend('linear').taper(0.01).filter('highpass', freq=1)
                f, t, spec = signal.stft(tr.data, tr.stats.sampling_rate, nperseg=NPERSEG,
                                         noverlap=NOVERLAP_SAMPLES, window='hann')
                freq_mask = (f >= FREQMIN) & (f <= FREQMAX)
                f_selected = f[freq_mask]
                t_mask = (t > 0) & (t < WINDOWS)
                t_selected = t[t_mask]
                spec_freq_selected = spec[freq_mask, :]
                spec_freq_t_selected = spec_freq_selected[:, t_mask]
                magnitude_spectrogram = np.abs(spec_freq_t_selected)
                # 对数缩放 (转换为分贝刻度)
                log_spectrogram = 10 * np.log10(magnitude_spectrogram + epsilon)
                # 使用最大值归一化
                min_val = np.min(log_spectrogram)
                max_val = np.max(log_spectrogram)
                if np.isclose(max_val, min_val):
                    normalized_spectrogram = np.zeros_like(log_spectrogram)
                else:
                    normalized_spectrogram = (log_spectrogram - min_val) / (max_val - min_val)
                return normalized_spectrogram
            else:
                return np.zeros((48, 40))

        stla = []
        stlo = []
        picktime = []
        spec_list = []
        distance_in_km_list = []
        azimuth_list = []
        N = nb_nodes
        for j in range(nb_nodes):
            spec_list_sta = []
            st, distance_in_meters, azimuth = Get_trace(ev_csv, j)
            if st is None:
                N = N - 1
                continue
            if st[0].stats.endtime - st[0].stats.starttime < WINDOWS:
                N = N - 1
                continue
            for tr in st:
                spec_list_sta.append(stft_spectrogram(tr))
            spec_array_sta = np.array(spec_list_sta)
            distance_in_km_list.append(distance_in_meters / 1000.0)
            azimuth_list.append(azimuth)
            stla.append(ev_csv['stla'].iloc[j])
            stlo.append(ev_csv['stlo'].iloc[j])
            picktime.append(UTCDateTime(ev_csv['picktime'].iloc[j]).timestamp)
            spec_list.append(spec_array_sta)
        return spec_list, stla, stlo, distance_in_km_list, azimuth_list, picktime, N
