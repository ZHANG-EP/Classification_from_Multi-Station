#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :Train.py
# @Time      :2025/9/3 8:09
# @Author    :ZHANG Yun in Rocket Force of University
# @Description: 
# @input     :
# @output    :
import torch
from Defaults import DEVICE
from sklearn.metrics import f1_score


def train(dataloader, model, optimizer, criterion):
    model.train()
    mean_loss = 0
    correct = 0
    total = 0
    all_labels = []
    all_preds = []
    for i, data in enumerate(dataloader):

        if torch.isnan(data.signal).any().item() or torch.isinf(data.signal).any().item():
            print("输入数据统计:")
            print(i)
            print(f"是否有NaN: {torch.isnan(data.signal).any().item()}")
            print(f"是否有Inf: {torch.isinf(data.signal).any().item()}")
            print(f"evid:{data.evid}")
            for j in range(data.signal.shape[0]):
                if torch.isnan(data.signal[j]).any().item() or torch.isinf(data.signal[j]).any().item():
                    print(f"第{j}个样本有问题")
                    print(data.pos[j])
                    print("zhangyun")

        data = data.to(DEVICE)
        optimizer.zero_grad()
        # Forward pass
        pred = model(data.event, data.pos, data.signal, data.azimuth_distance, data.edge_index, data.edge_weight,
                     data.batch)
        loss = criterion(pred, data.y)
        loss.backward()
        optimizer.step()
        mean_loss += loss.item()
        _, predicted = torch.max(pred.data, 1)
        all_labels.append(data.y)
        all_preds.append(predicted)
        total += data.y.size(0)
        correct += (predicted == data.y).sum().item()
    accuracy = correct / total * 100
    labels_np = torch.cat(all_labels, dim=0).cpu().numpy()
    preds_np = torch.cat(all_preds, dim=0).cpu().numpy()
    current_f1 = f1_score(labels_np, preds_np, average="macro")
    return mean_loss / len(dataloader), accuracy, current_f1


@torch.no_grad()
def validation(dataloader, model, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_labels = []
    all_preds = []
    for data in dataloader:
        data = data.to(DEVICE)
        pred = model(data.event, data.pos, data.signal, data.azimuth_distance, data.edge_index, data.edge_weight,
                     data.batch)
        loss = criterion(pred, data.y)
        total_loss += loss.item()
        _, predicted = torch.max(pred.data, 1)
        all_labels.append(data.y)
        all_preds.append(predicted)
        total += data.y.size(0)
        correct += (predicted == data.y).sum().item()
    accuracy = 100 * correct / total
    labels_np = torch.cat(all_labels, dim=0).cpu().numpy()
    preds_np = torch.cat(all_preds, dim=0).cpu().numpy()
    current_f1 = f1_score(labels_np, preds_np, average="macro")
    return total_loss / len(dataloader), accuracy, current_f1
