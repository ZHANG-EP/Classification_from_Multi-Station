from tqdm import tqdm
import torch
from torch_geometric.loader import DataLoader
import pandas as pd
from Graph_Datasets import GraphDataset, GraphDataset_Multi
from torch_geometric.transforms import KNNGraph
from Defaults import KNN_K, DATA_AUGUMENT
from Defaults import Train_End_Time, Valid_End_Time, EPOCHES, LEARNING_RATE, WEIGHT_DECAY, DEVICE
from utils import NormalizeTargets_LAT_LONG, get_split_indices
import os
import GraphNet_Signal, GraphNet_Signal_Station, GraphNet_Signal_Event, GraphNet_Signal_Event_Station
import Train
from collections import Counter


# 计算每个数据集中标签的分布
def count_labels(data_subset, subset_name):
    labels = [data.y.item() for data in data_subset]  # 获取所有图的标签
    label_counts = Counter(labels)
    print(f"{subset_name}: 总共 {len(data_subset)} 个图")
    print(f"  标签 0: {label_counts.get(0, 0)} 个图,占比：{label_counts.get(0, 0) / len(data_subset) * 100}%")
    print(f"  标签 1: {label_counts.get(1, 0)} 个图,占比：{label_counts.get(1, 0) / len(data_subset) * 100}%")

    node_counts = {0: 0, 1: 0}  # 初始化节点计数

    for data in data_subset:
        label = data.y.item()
        node_counts[label] += data.num_nodes  # 累加该图的节点数
    print(
        f"节点总数: {node_counts.get(0, 0)} 个, 平均每图: {node_counts.get(0, 0) / label_counts.get(0, 1) :.2f} 个节点")
    print(
        f"节点总数: {node_counts.get(1, 0)} 个, 平均每图: {node_counts.get(1, 0) / label_counts.get(1, 0):.2f} 个节点")


if __name__ == "__main__":
    csv = pd.read_csv(r'./input/download_dataset_updated.csv')
    csv = csv[csv["Typ"] != "ex"].reset_index(drop=True)
    csv = csv[csv['chan'].str.endswith('Z', na=False)].reset_index(drop=True)
    dataset = GraphDataset_Multi(root="./Ulta_datasets",
                                 pre_transform=KNNGraph(k=KNN_K, loop=False, force_undirected=True),
                                 csv=csv, augument=True)

    count_labels(dataset, "完整数据集")

    # 训练集: 13688 个图    验证集: 1810 个图     测试集: 1124 个图
    train_idx, val_idx, test_idx = get_split_indices(dataset, Train_End_Time, Valid_End_Time)

    count_labels(dataset[train_idx], "训练集")
    count_labels(dataset[val_idx], "验证集")
    count_labels(dataset[test_idx], "测试集")

    # # 创建数据加载器
    # 训练集： 7618（55.65%） 地震                6070（44.35%） 爆破
    # 增强：36194（50.8785%） 地震                34944 （49.12142%） 爆破
    train_loader = DataLoader(NormalizeTargets_LAT_LONG(dataset[train_idx]), batch_size=128, shuffle=True)
    # 验证集： 932（51.49%） 地震                 878（48.51%） 爆破
    val_loader = DataLoader(NormalizeTargets_LAT_LONG(dataset[val_idx]), batch_size=128, shuffle=False)


    # 测试集： 260（23.13%） 地震                 864（76.87%） 爆破
    # test_loader = DataLoader(NormalizeTargets_LAT_LONG(dataset[test_idx]), batch_size=32, shuffle=False)

    def Train_main(model, train_loader, val_loader, model_path_save, train_losses_acc_path, lr):
        print(lr)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
        train_losses = []
        train_f1s = []
        val_losses = []
        val_f1s = []
        train_accuracies = []
        val_accuracies = []
        best_f1 = 0.0
        best_epoch = 0
        # Create the folder if it doesn't exist
        os.makedirs(os.path.dirname(model_path_save), exist_ok=True)
        nb_epoch = tqdm(range(EPOCHES))
        for epoch in nb_epoch:
            train_loss, train_acc, train_f1 = Train.train(train_loader, model, optimizer, criterion)
            val_loss, val_acc, val_f1 = Train.validation(val_loader, model, criterion)
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accuracies.append(train_acc)
            val_accuracies.append(val_acc)
            train_f1s.append(train_f1)
            val_f1s.append(val_f1)

            if val_f1 > best_f1:
                best_f1 = val_f1
                best_epoch = epoch
                torch.save(model.state_dict(), model_path_save)
            nb_epoch.set_postfix_str(
                f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f},"
                f"Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%"
                f"Train F1: {train_f1: .4f}, Val F1: {val_f1: .4f}"
            )
        print("\n" + "=" * 80)
        print("训练结果汇总:")
        print("=" * 80)
        for epoch, (t_loss, v_loss, t_acc, v_acc) in enumerate(
                zip(train_losses, val_losses, train_accuracies, val_accuracies)):
            print(f"Epoch {epoch:3d}: Train Loss={t_loss:.4f}, Val Loss={v_loss:.4f}, "
                  f"Train Acc={t_acc:.2f}%, Val Acc={v_acc:.2f}%")

        print(f"\n最佳模型在 Epoch {best_epoch}: Val f1={best_f1:.4f}")
        reselt = pd.DataFrame(columns=['train_loss', 'val_loss', 'train_acc', 'valid_acc', "train_f1", "val_f1"])
        reselt['train_loss'] = train_losses
        reselt['val_loss'] = val_losses
        reselt['train_acc'] = train_accuracies
        reselt['valid_acc'] = val_accuracies
        reselt['train_f1'] = train_f1s
        reselt["val_f1"] = val_f1s
        reselt.to_csv(train_losses_acc_path, index=False)


    # 数据增强: Link_All F1: 0.9867
    # 数据增强: Link_Five F1: 0.9867
    for i, lr in enumerate(LEARNING_RATE):
        print(f"正在用学习率{lr}训练")
        Train_main(GraphNet_Signal.GraphNet().to(DEVICE), train_loader, val_loader,
                   model_path_save=f"./model/Signal/Model_Signal_lr={lr}.pth",
                   train_losses_acc_path=f"./result/Signal/Train_Result_Signal_lr={lr}.csv", lr=lr)
