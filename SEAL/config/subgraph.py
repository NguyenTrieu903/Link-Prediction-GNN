from operator import itemgetter

import networkx as nx
import numpy as np
from sklearn import metrics
from tqdm import tqdm

from SEAL.utils import utils


def link2subgraph(positive, negative, nodes_size, test_ratio, hop, network_type, max_neighbors=100):
    """
    :param positive: ndarray, from 'load_data', all positive edges
    :param negative: ndarray, from 'load_data', all negative edges
    :param nodes_size: int, scalar, nodes size in the network
    :param test_ratio: float, scalar, proportion of the test set
    :param hop: option: 0, 1, 2, ..., or 'auto'
    :param network_type: directed or undirected
    :param max_neighbors:
    :return:
    """
    print("extract enclosing subgraph...")
    test_size = int(len(positive) * test_ratio)
    train_pos, test_pos = positive[:-test_size], positive[-test_size:]
    train_neg, test_neg = negative[:-test_size], negative[-test_size:]

    A = np.zeros([nodes_size, nodes_size], dtype=np.uint8)
    # create adjacency matrix of graph
    A[train_pos[:, 0], train_pos[:, 1]] = 1
    if network_type == 0:
        A[train_pos[:, 1], train_pos[:, 0]] = 1

    def calculate_auc(scores, test_pos, test_neg):
        pos_scores = scores[test_pos[:, 0], test_pos[:, 1]]
        neg_scores = scores[test_neg[:, 0], test_neg[:, 1]]
        s = np.concatenate([pos_scores, neg_scores])
        y = np.concatenate([np.ones(len(test_pos), dtype=np.int8), np.zeros(len(test_neg), dtype=np.int8)])
        assert len(s) == len(y)
        auc = metrics.roc_auc_score(y_true=y, y_score=s)
        return auc

    # determine the h value
    # doan code dung de tinh toan do tuong dong giua cac nut trong do thi dua tren thong tin cau truc.
    # 2 do tuong dong duoc tinh o day la: cn(Common Neighbors) va aa(Adamic-Adar).... Sau do so sanh xe auc cua phuong phap nao lon hon thi su dung phuong phap do
    if hop == "auto":
        def cn():
            return np.matmul(A, A)

        def aa():
            A_ = A / np.log(A.sum(axis=1))
            A_[np.isnan(A_)] = 0
            A_[np.isinf(A_)] = 0
            return A.dot(A_)

        cn_scores, aa_scores = cn(), aa()
        cn_auc = calculate_auc(cn_scores, test_pos, test_neg)
        aa_auc = calculate_auc(aa_scores, test_pos, test_neg)
        if cn_auc > aa_auc:
            print("cn(first order heuristic): %f > aa(second order heuristic) %f." % (cn_auc, aa_auc))
            hop = 1
        else:
            print("aa(second order heuristic): %f > cn(first order heuristic) %f. " % (aa_auc, cn_auc))
            hop = 2

    print("hop = %s." % hop)

    # tao mot do thi moi G de tinh toan, di tim cac subgraph
    # extract the subgraph for (positive, negative)
    G = nx.Graph() if network_type == 0 else nx.DiGraph()
    G.add_nodes_from(set(positive[:, 0]) | set(positive[:, 1]) | set(negative[:, 0]) | set(negative[:, 1]))
    # G.add_nodes_from(set(sum(positive.tolist(), [])) | set(sum(negative.tolist(), [])))
    G.add_edges_from(train_pos)

    graphs_adj, labels, vertex_tags, node_size_list, sub_graphs_nodes = [], [], [], [], []
    for graph_label, data in enumerate([negative, positive]):
        print("for %s. " % "negative" if graph_label == 0 else "positive")
        for node_pair in tqdm(data):
            sub_nodes, sub_adj, vertex_tag = extract_subgraph(node_pair, G, A, hop, network_type, max_neighbors)
            graphs_adj.append(sub_adj)
            vertex_tags.append(vertex_tag)
            node_size_list.append(len(vertex_tag))
            sub_graphs_nodes.append(sub_nodes)
    assert len(graphs_adj) == len(vertex_tags) == len(node_size_list)
    # dung de tao ma tran nhan cho cac canh trong do thi. tao thanh ma tran co 1 cot va so hang la tong so cac canh trong do thi.
    # Ket qua la mot ma tran nhan, trong do moi hang dai dien cho 1 canh trong do thi va mot nhan tuong ung.
    labels = np.concatenate([np.zeros(len(negative), dtype=np.uint8), np.ones(len(positive), dtype=np.uint8)]).reshape(
        -1, 1)

    # vertex_tags_set = list(set(sum(vertex_tags, [])))
    vertex_tags_set = set()
    for tags in vertex_tags:
        vertex_tags_set = vertex_tags_set.union(set(tags))
    vertex_tags_set = list(vertex_tags_set)
    tags_size = len(vertex_tags_set)
    print("tight the vertices tags.")
    # kiem tra xem tat ca cac phan tu trong vertex_tags_set co tao thanh mot chuoi so nguyen lien tuc tu 0 den len(vertex_tags_set-1) hay khong.
    # Dieu nay dung de dam bao tinh day du va dung dan cua cac nhan duoc gan cho cac nut trong do thi.
    if set(range(len(vertex_tags_set))) != set(vertex_tags_set):
        vertex_map = dict([(x, vertex_tags_set.index(x)) for x in vertex_tags_set])
        for index, graph_tag in tqdm(enumerate(vertex_tags)):
            vertex_tags[index] = list(itemgetter(*graph_tag)(vertex_map))
    return graphs_adj, labels, vertex_tags, node_size_list, sub_graphs_nodes, tags_size


def extract_subgraph(node_pair, G, A, hop, network_type, max_neighbors):
    """
    :param node_pair:  (vertex_start, vertex_end)
    :param G:  nx object from the positive edges
    :param A:  equivalent to the G, adj matrix of G
    :param hop:
    :param network_type:
    :param max_neighbors:
    :return:
        sub_graph_nodes: use for select the embedding feature
        sub_graph_adj: adjacent matrix of the enclosing sub-graph
        vertex_tag: node type information from the labeling algorithm
    """
    sub_graph_nodes = set(node_pair)
    nodes = list(node_pair)
    nodes = list(nodes)
    for i in range(int(hop)):
        np.random.shuffle(nodes)
        for node in nodes:
            neighbors = list(nx.neighbors(G, node))
            if len(sub_graph_nodes) + len(neighbors) < max_neighbors:
                sub_graph_nodes = sub_graph_nodes.union(neighbors)
            else:
                np.random.shuffle(neighbors)
                sub_graph_nodes = sub_graph_nodes.union(neighbors[:max_neighbors - len(sub_graph_nodes)])
                break
        nodes = sub_graph_nodes - set(nodes)
    sub_graph_nodes.remove(node_pair[0])
    if node_pair[0] != node_pair[1]:
        sub_graph_nodes.remove(node_pair[1])
    sub_graph_nodes = [node_pair[0], node_pair[1]] + list(sub_graph_nodes)
    sub_graph_adj = A[sub_graph_nodes, :][:, sub_graph_nodes]
    sub_graph_adj[0][1] = sub_graph_adj[1][0] = 0

    # labeling(coloring/tagging)
    vertex_tag = utils.node_labeling(sub_graph_adj, network_type)
    return sub_graph_nodes, sub_graph_adj, vertex_tag
