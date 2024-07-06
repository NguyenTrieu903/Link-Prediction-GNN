import pandas as pd
from torch_geometric.data import Data

# from utils import get_ei2, random_split_edges
import torch
from torch_geometric.utils import to_undirected, is_undirected, negative_sampling, add_self_loops
# from utils import double, degree
from TwoWL.utils import *
import constant

class dataset:
    def __init__(self, x, na, ei, ea, pos1, y, ei2):
        self.x = x
        self.na = na
        self.ei = ei
        self.ea = ea
        self.pos1 = pos1
        self.y = y
        self.ei2 = ei2


class BaseGraph:
    def __init__(self, x, node_attr, edge_pos, edge_neg, num_pos, num_neg, pattern):
        self.x = x
        self.node_attr = node_attr
        self.edge_pos = edge_pos
        self.edge_neg = edge_neg
        self.num_pos = num_pos
        self.num_neg = num_neg
        self.num_nodes = x.shape[0]
        self.max_x = None
        self.pattern = pattern

    def toString(self):
        return f"BaseGraph object:\n\
        - x: {self.x}\n\
        - node_attr: {self.node_attr}\n\
        - edge_pos: {self.edge_pos}\n\
        - edge_neg: {self.edge_neg}\n\
        - num_pos: {self.num_pos}\n\
        - num_neg: {self.num_neg}\n\
        - num_nodes: {self.num_nodes}\n\
        - max_x: {self.max_x}\n\
        - pattern: {self.pattern}"
    def preprocess(self):
        self.edge_indexs = [
            self.edge_pos[:, :self.num_pos[0]],
            self.edge_pos[:, :self.num_pos[1]],
            self.edge_pos[:, :self.num_pos[2]]
        ]

        self.edge_attrs = [
            torch.ones_like(self.edge_indexs[i][0], dtype=torch.float)
            for i in range(3)
        ]
        '''if self.attr == None else [
            self.edge_attr[0],
            self.edge_attr[1],
            self.edge_attr[2]
        ]
        '''
        pos_edges = [
            self.edge_pos[:, :self.num_pos[0]],
            self.edge_pos[:, :self.num_pos[1]],
            self.edge_pos[:, :self.num_pos[2]]
        ]
        neg_edges = [
            self.edge_neg[:, :self.num_neg[0]],
            self.edge_neg[:, :self.num_neg[1]],
            self.edge_neg[:, :self.num_neg[2]]
        ]

        # dim = 1 sẽ nối theo chiều từ trái sang phải đọc lần lượt từng tensor rồi nối lại
        # Thêm neg_edges[0] vào đầu danh sách các cạnh dự đoán để đảm bảo rằng tập dữ liệu dự đoán bao gồm cả cạnh dương và cạnh âm
        pred_edges = [neg_edges[0]] + [
            torch.cat((pos_edges[i], neg_edges[i]), dim=1)
            for i in range(1, 3)
        ]

        # đoạn code này nối edge_indexs và pred_edges theo chiều hàng.
        # nhằm mục đích kết hợp các cạnh từ đồ thị gốc và các cạnh dự đoán, chuyển đổi chúng, và lưu trữ trong danh sách mới
        # pos1s = 'positive edges set 1'
        self.pos1s = [
            torch.cat((self.edge_indexs[i].t(), pred_edges[i].t()), dim=0)
            for i in range(3)
        ]

        # Dùng để tạo các nhãn cho các cạnh. Cạnh positive được gán giá trị 1 và cạnh neg được gán giá trị 0
        self.ys = [torch.zeros((neg_edges[0].shape[1], 1),
                                 device=self.edge_pos.device)]+[
            torch.cat((torch.ones((pos_edges[i].shape[1], 1),
                                  dtype=torch.float,
                                  device=self.edge_pos.device),
                       torch.zeros((neg_edges[i].shape[1], 1),
                                   dtype=torch.float,
                                   device=self.edge_pos.device)))
            for i in range(1, 3)
        ]
        """
            ei2: edge_index_2 hoặc edge_index_extended
            ei2 có thể biểu thị một phiên bản mở rộng hoặc biến đổi của edge_index, có thể bao gồm thông tin bổ sung hoặc cấu trúc đặc biệt nào đó.
            Trong một số trường hợp, ei2 có thể chứa các cặp chỉ số cạnh cho một tập hợp cạnh mở rộng, như các cạnh dự đoán hoặc các cạnh đã được xử 
            lý qua một hàm cụ thể.
        """
        if self.pattern == '2wl_l':
            self.ei2s = [
                get_ei2(self.num_nodes, self.edge_indexs[0], pred_edges[0]),
                get_ei2(self.num_nodes, self.edge_indexs[1], pred_edges[1]),
                get_ei2(self.num_nodes, self.edge_indexs[2], pred_edges[2])
            ]


    def split(self, split: int):
        return self.x[split], self.node_attr, self.edge_indexs[split], self.edge_attrs[
            split], self.pos1s[split], self.ys[split], self.ei2s[split]

    def setPosDegreeFeature(self):
        self.x_backup = [
                      degree(self.edge_indexs[0], self.num_nodes) for i in range(0, 2)
                  ] + [
                      degree(self.edge_indexs[1], self.num_nodes) for i in range(2, 3)
                  ]
        self.x = [
                     degree(self.edge_indexs[0], self.num_nodes)
                 ] + [
                     degree(self.edge_indexs[1], self.num_nodes)
                 ] + [
                     degree(self.edge_indexs[2], self.num_nodes)
                 ]
        self.max_x = max([torch.max(_).item() for _ in self.x])

    def to(self, device):
        self.x = self.x.to(device)
        self.edge_pos = self.edge_pos.to(device)
        self.edge_neg = self.edge_neg.to(device)
        return self


def load_dataset(pattern, trn_ratio=0.8, val_ratio=0.05, test_ratio=0.1):
        split_edge = load({
            "data_name": "fb-pages-food",
            "train_name": None,
            "test_name": None,
            "val_ratio": val_ratio,
            "test_ratio": test_ratio,
            "max_train_num": 1000000000
        })

        train_pos = double(split_edge['train']['edge'])
        train_neg = double(split_edge['train']['edge_neg'])
        val_pos = double(split_edge["valid"]["edge"])
        val_neg = double(split_edge["valid"]["edge_neg"])
        test_pos = double(split_edge["test"]["edge"])
        test_neg = double(split_edge["test"]["edge_neg"])

        edge_pos = torch.cat((train_pos, val_pos, test_pos), dim=-1)
        edge_neg = torch.cat((train_neg, val_neg, test_neg), dim=-1)

        num_pos = torch.tensor(
            [train_pos.shape[1], val_pos.shape[1], test_pos.shape[1]])
        num_neg = torch.tensor(
            [train_neg.shape[1], val_neg.shape[1], test_neg.shape[1]])
        n_node = max(torch.max(edge_pos), torch.max(edge_neg)) + 1

        x = torch.zeros((n_node, 0))
        bg =  BaseGraph(x, None, edge_pos, edge_neg, num_pos, num_neg, pattern)
        return bg


def load(args):
    print("run")
    df = pd.read_csv(constant.PATH_CSV_EDGES, header=None)  # Assuming there are no column names in the CSV
    source_nodes = df[0]

    target_nodes = df[1]

    # Create tensors for source and target nodes
    edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
    row, col = edge_index[0], edge_index[1]
    edge_index = torch.stack((row, col))

    data = Data(edge_index=edge_index)
    neg_pool_max = False
    split_edge = do_edge_split(data, args["val_ratio"], args["test_ratio"], neg_pool_max)
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

    print("data.train_pos_edge_index", data.train_pos_edge_index.shape)
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
    print("split_edge['train']['edge']", split_edge['train']['edge'].size())
    return split_edge