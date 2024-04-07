'''
使用方法
train_pos, train_neg, test_pos, test_neg = load({
            "data_name":
            "Celegans",
            "train_name":
            None,
            "test_name":
            None,
            "test_ratio":
            0.1,
            "max_train_num":
            1000000000
        })
train_pos, train_neg, test_pos, test_neg 分别为训练集中正边，负边，测试集中正边，负边
'''
import os.path as osp
import numpy as np
import torch
import scipy.io as sio
import scipy.sparse as ssp
from torch_geometric.utils import negative_sampling, add_self_loops
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from ogb.linkproppred import PygLinkPropPredDataset
from utils import random_split_edges
from tqdm import tqdm
import pandas as pd


def to_undirected(edge_index):
    n = edge_index.max() + 1
    A = torch.zeros((n, n), dtype=torch.bool)
    index = torch.empty((2, edge_index.shape[1] // 2), dtype=torch.long)
    t = 0
    for i in tqdm(range(edge_index.shape[1])):
        p, q = edge_index[0, i], edge_index[1, i]
        if not A[q, p]:
            index[0, t] = p
            index[1, t] = q
            A[p, q] = 1
            t += 1
    return index


def tail(edge_index, s):
    mask = (edge_index[0, :] < s) & (edge_index[1, :] < s)
    return edge_index[:, mask]


def load(args):
    # row: doc tu file csv
    # col: doc tu file csv
    # if args["data_name"] == "fb-pages-food":
        print("run")
        df = pd.read_csv('dataset/fb-pages-food.csv', header=None)  # Assuming there are no column names in the CSV
        # Split the data into source and target nodes
        source_nodes =df[0]


        target_nodes = df[1]

        # Create tensors for source and target nodes
        edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
        row, col = edge_index[0], edge_index[1]
        edge_index = torch.stack((row, col))

        data = Data(edge_index=edge_index)
        neg_pool_max = False
        split_edge = do_edge_split(data, args["val_ratio"], args["test_ratio"], neg_pool_max)
        # print(split_edge)
        # size = data.x.shape[0]//10
        # # data.num_nodes = size
        # # data.x = data.x[:size,:]
        # # x = x[:size,:]
        # # data.edge_index = tail(data.edge_index, size)
        # split_edge['train']['edge'] = tail(split_edge['train']['edge'], size)
        # split_edge['train']['edge_neg'] = negative_sampling(
        #     data.edge_index,
        #     num_nodes=data.num_nodes,
        #     num_neg_samples=split_edge['train']['edge'].shape[1])
        # split_edge['valid']['edge'] = tail(split_edge['valid']['edge'], size)
        # split_edge['valid']['edge_neg'] = tail(split_edge['valid']['edge_neg'], size)
        # split_edge['test']['edge'] = tail(split_edge['test']['edge'], size)
        # split_edge['test']['edge_neg'] = tail(split_edge['test']['edge_neg'], size)

        return split_edge


def do_edge_split(data, val_ratio=0.05, test_ratio=0.1, neg_pool_max=False):
    data = random_split_edges(data, val_ratio, test_ratio)
    edge_index, _ = add_self_loops(data.train_pos_edge_index)

    if not neg_pool_max:
        data.train_neg_edge_index = negative_sampling(
            edge_index,
            num_nodes=data.num_nodes,
            num_neg_samples=data.train_pos_edge_index.shape[1])

    else:
        data.train_neg_edge_index = negative_sampling(
            edge_index,
            num_nodes=data.num_nodes,
            num_neg_samples=data.train_pos_edge_index.shape[1])

    print(data.train_neg_edge_index.shape)
    data.val_neg_edge_index = negative_sampling(
        torch.cat((edge_index, data.val_pos_edge_index), dim=-1),
        num_nodes=data.num_nodes,
        num_neg_samples=data.val_pos_edge_index.shape[1])
    data.test_neg_edge_index = negative_sampling(
        torch.cat(
            (edge_index, data.val_pos_edge_index, data.test_pos_edge_index),
            dim=-1),
        num_nodes=data.num_nodes,
        num_neg_samples=data.test_pos_edge_index.shape[1])

    split_edge = {'train': {}, 'valid': {}, 'test': {}}
    split_edge['train']['edge'] = data.train_pos_edge_index
    split_edge['train']['edge_neg'] = data.train_neg_edge_index
    split_edge['valid']['edge'] = data.val_pos_edge_index
    split_edge['valid']['edge_neg'] = data.val_neg_edge_index
    split_edge['test']['edge'] = data.test_pos_edge_index
    split_edge['test']['edge_neg'] = data.test_neg_edge_index
    return split_edge
