#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :Pred.py
# @Time      :2025/9/3 19:21
# @Author    :ZHANG Yun in Rocket Force of University
# @Description: 
# @input     :
# @output    :
import torch
from torch_geometric.loader import DataLoader
import pandas as pd
import GraphNet
from collections import Counter

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def Pred(dataloader, model):
    model.eval()
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []
    evid = []
    evla = []
    evlo = []
    evdp = []
    prob = []

    for data in dataloader:
        data = data.to(DEVICE)
        # pred = model(data.event, data.pos, data.signal, data.azi_dis, data.edge_index, data.edge_weight,
        #              data.batch)
        pred = model(data.signal, data.edge_index, data.edge_weight, data.batch)
        score = torch.softmax(pred, dim=1)[0, 1].item()
        prob.append(score)
        _, predicted = torch.max(pred.data, 1)
        total += data.y.size(0)
        correct += (predicted == data.y).sum().item()
        # data.event[:, :2] = test_data.denormalize(data.event[:, :2])
        all_predictions.extend(predicted.cpu().numpy())
        all_labels.extend(data.y.cpu().numpy())
        evid.extend(data.evid.cpu().numpy()[0])
        evla.extend(data.event[:, 0].cpu().numpy())
        evlo.extend(data.event[:, 1].cpu().numpy())
        evdp.extend(data.event[:, 2].cpu().numpy())
    results_df = pd.DataFrame({
        'evid': evid,
        'evla': evla,
        'evlo': evlo,
        'evdp': evdp,
        'true_label': all_labels,
        'predicted_label': all_predictions,
        'prob': prob
    })
    accuracy = 100 * correct / total
    return accuracy, results_df


def count_labels(data_subset, subset_name):
    class_node_count = Counter()
    class_graph_count = Counter()
    for data in data_subset:
        label = data.y.item()
        class_graph_count[label] += 1
        num_nodes = data.pos.size(0)
        class_node_count[label] += num_nodes

    print(f"{subset_name}: total {len(data_subset)} graphs")
    print(
        f"  Label 0: {class_graph_count[0]} graphs, ({class_graph_count[0] / len(data_subset) * 100}%), nodes：{class_node_count[0]}")
    print(
        f"  Label 1: {class_graph_count[1]} graphs, ({class_graph_count[1] / len(data_subset) * 100}%), nodes：{class_node_count[1]}")


if __name__ == "__main__":
    test_data = torch.load('./test_data/test_data.pt')
    count_labels(test_data, "Test Data")

    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)
    model = GraphNet.GraphNet().to(DEVICE)
    model.load_state_dict(torch.load(f"./../model/Model.pth", map_location=DEVICE))
    accuracy, results_df = Pred(test_loader, model)

    results_df.to_csv(f"./result/Test_result.csv", index=False)
    print(f"Test Accuracy: {accuracy:.2f}%")
