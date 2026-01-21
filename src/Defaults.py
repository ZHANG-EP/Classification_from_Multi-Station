#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :Defaults.py
# @Time      :2025/9/3 8:05
# @Author    :ZHANG Yun in Rocket Force of University
# @Description: 
# @input     :
# @output    :
import numpy as np
import torch
import pickle
from obspy import UTCDateTime

# --------------------------------------------------------------
# -------------------Results Save-------------------------------
# --------------------------------------------------------------
Model_Path_Save_Sta_Event = "model/GraphNet_Sta_Event.pth"
Train_Losses_Acc_Sta_Event = "./result/GraphNet_Sta_Event.csv"

Model_Path_Save_Azi_Distance = "model/GraphNet_Azi_Distance.pth"
Train_Losses_Acc_Azi_Distance = "./result/GraphNet_Azi_Distance.csv"

Model_Path_Save_Only_Signal = "model/GraphNet_Only_Signal.pth"
Train_Losses_Acc_Only_Signal = "./result/GraphNet_Only_Signal.csv"

Model_Path_Save_Signal_Sta_Azi_Dis = "model/GraphNet_Signal_Sta_Azi_Dis.pth"
Train_Losses_Acc_Signal_Sta_Azi_Dis = "./result/GraphNet_Signal_Sta_Azi_Dis.csv"

Model_Path_Save_Signal_Event = "model/GraphNet_Signal_Event.pth"
Train_Losses_Acc_Signal_Event = "./result/GraphNet_Signal_Event.csv"

Model_Path_Save_Signal_Event_Without_Depth = "model/GraphNet_Signal_Event_Without_Depth.pth"
Train_Losses_Acc_Signal_Event_Without_Depth = "./result/GraphNet_Signal_Event_Without_Depth.csv"

Model_Path_Save_Signal_Event_Without_Depth_STA = "model/GraphNet_Signal_Event_Without_Depth_STA.pth"
Train_Losses_Acc_Signal_Event_Without_Depth_STA = "./result/GraphNet_Signal_Event_Without_Depth_STA.csv"

Model_Path_Save_Signal_STA = "model/GraphNet_Signal_Sta.pth"
Train_Losses_Acc_Signal_STA = "./result/GraphNet_Signal_Sta.csv"
# --------------------------------------------------------------
# -------------------Data Split-------------------------------
# --------------------------------------------------------------
Train_End_Time = UTCDateTime("2019-01-01T00:00:00Z")
Valid_End_Time = UTCDateTime("2020-01-01T00:00:00Z")
Train_Df_Path = r"./input/train.csv"
Valid_Df_Path = r"./input/valid.csv"
Test_Df_Path = r"./input/test.csv"

# ------------------------------------------------------------
# ------------------Data Augument Parameters------------------
# ------------------------------------------------------------
NUM_WORKS = 12
DATA_AUGUMENT = False
Amp_Normalize = True
Train_Name = "./Train_Data_Namp"
Valid_Name = "./Valid_Data_Namp"
Test_Name = "./Test_Data_Namp"
EQ_Augument_Samples = 4
QB_Augument_Samples = 3
KNN_K = 300
SAC_ROOT = r"../00 SAC"
# -------------------STFT 参数---------------------------------
NPERSEG = 250
NOVERLAP_SAMPLES = int(0.12 * NPERSEG)
epsilon = 1e-10
FREQMIN = 1.0
FREQMAX = 20.0
# ------------------Learning Parameters-----------------------

BATCH_SIZE = 500
# LEARNING_RATE = 0.001
LEARNING_RATE = [0.0001, 0.0005, 0.001]
WEIGHT_DECAY = 1e-4
EPOCHES = 100
# ------------------------------------------------------------
# ------------------Data Preprocess---------------------------
# ------------------------------------------------------------
label_mapping = {
    'le': 0,
    'qb': 1
}
SAMPLING_RATE = 100
WINDOWS = 90
Freqmin = 1.0
Freqmax = 20.0
# -------------------------------------------------------------
# ---------------------Fixed Random Seed----------------------
# -------------------------------------------------------------
RAND_SEED = 42
RNG = np.random.default_rng(RAND_SEED)
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# -------------------------------------------------------------
# -------------------Normalize Regions-------------------------
# -------------------------------------------------------------
MIN_LAT = 32
MAX_LAT = 45
MIN_LONG = -115
MAX_LONG = -108

MIN_AZI = 0.0
MAX_AZI = 360.0
MIN_DISTANCE = 0.0
MAX_DISTANCE = 400.0
