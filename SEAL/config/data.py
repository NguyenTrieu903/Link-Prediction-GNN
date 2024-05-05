import networkx as nx
import numpy as np
import pandas as pd
from gensim.models import Word2Vec

import constant
from node2vec.src import node2vec


def load_data(network_type):
    """
    :param data_name:
    :param network_type: use 0 and 1 stands for undirected or directed graph, respectively
    :return:
    """
    print("load data...")
    positive_df = pd.read_csv(constant.PATH_EDGES, delimiter=',', dtype=int)
    positive = positive_df.to_numpy()

    # sample negative
    G = nx.Graph() if network_type == 0 else nx.DiGraph()
    G.add_edges_from(positive)
    # print(nx.info(G))
    negative_all = list(nx.non_edges(G))
    np.random.shuffle(negative_all)
    negative = np.asarray(negative_all[:len(positive)])
    print("positve examples: %d, negative examples: %d." % (len(positive), len(negative)))
    np.random.shuffle(positive)
    if np.min(positive) == 1:
        positive -= 1
        negative -= 1
    return positive, negative, len(G.nodes())

def learning_embedding(positive, negative, network_size, test_ratio, dimension, network_type, negative_injection=True):
    """
    :param positive: ndarray, from 'load_data', all positive edges
    :param negative: ndarray, from 'load_data', all negative edges
    :param network_size: scalar, nodes size in the network
    :param test_ratio: proportion of the test set
    :param dimension: size of the node2vec
    :param network_type: directed or undirected
    :param negative_injection: add negative edges to learn word embedding
    :return:
    """
    print("learning embedding...")
    # used training data only for embedding process
    test_size = int(test_ratio * positive.shape[0])
    train_posi, train_nega = positive[:-test_size], negative[:-test_size]
    # negative injection
    A = nx.Graph() if network_type == 0 else nx.DiGraph()

    # So if train_posi is an array containing the edges of the graph,
    # the line of code will add these edges to graph A and assign a weight of 1 to each edge.
    A.add_weighted_edges_from(
        np.concatenate([train_posi, np.ones(shape=[train_posi.shape[0], 1], dtype=np.int8)], axis=1))
    if negative_injection:
        A.add_weighted_edges_from(
            np.concatenate([train_nega, np.ones(shape=[train_nega.shape[0], 1], dtype=np.int8)], axis=1))
    line_graph = nx.line_graph(A)
    # node2vec
    G = node2vec.Graph(A, is_directed=False if network_type == 0 else True, p=1, q=1)
    G.preprocess_transition_probs()
    walks = G.simulate_walks(num_walks=10, walk_length=80)
    walks = [list(map(str, walk)) for walk in walks]
    model = Word2Vec(walks, size=dimension, window=10, min_count=0, sg=1, workers=8, iter=1)
    wv = model.wv
    embedding_feature, empty_indices, avg_feature = np.zeros([network_size, dimension]), [], 0
    for i in range(network_size):
        if str(i) in wv:
            embedding_feature[i] = wv.word_vec(str(i))
            # print("embedding_feature[{}]: {}".format(i, embedding_feature[i]))
            avg_feature += wv.word_vec(str(i))
        else:
            empty_indices.append(i)
    embedding_feature[empty_indices] = avg_feature / (network_size - len(empty_indices))
    print("embedding feature shape: ", embedding_feature.shape)
    return embedding_feature

def create_input_for_gnn_fly(graphs_adj, labels, vertex_tags, node_size_list, sub_graphs_nodes,
                             embedding_feature, explicit_feature, tags_size):
    print("create input for gnn on fly, (skipping I/O operation)")
    # graphs, nodes_size_list, labels = data["graphs"], data["nodes_size_list"], data["labels"]

    # 1 - prepare Y
    # dung de tao ma tran nhan Y trong do moi phan tu co gia tri 1 neu tuong ung voi mot canh co nhan 1, va co gia tri 0 neu tuong ung voi 1 canh co nhan 0
    Y = np.where(np.reshape(labels, [-1, 1]) == 1, 1, 0)
    print("positive examples: %d, negative examples: %d." % (np.sum(Y == 0), np.sum(Y == 1)))
    # 2 - prepare A_title
    # graphs_adj is A_title in the formular of Graph Convolution layer
    # add eye to A_title
    # code dung de them ma tran don vi vao moi ma tran ke. np.eye dung de tao ma tran don vi
    for index, x in enumerate(graphs_adj):
        graphs_adj[index] = x + np.eye(x.shape[0], dtype=np.uint8)
    # 3 - prepare D_inverse
    D_inverse = []
    for x in graphs_adj:
        # su dung de tinh ma tran nghich dao cua ma tran duong cheo. np.sum(x, axis=1) -> tinh tong moi hang cua ma tran
        # np.diag -> tao mot ma tran duong cheo tu mang bang cach su dung np.diag()
        # np.linalg.inv -> tinh ma tran nghich dao cua ma tran duong cheo nay.
        D_inverse.append(np.linalg.inv(np.diag(np.sum(x, axis=1))))
    # 4 - prepare X
    X, initial_feature_channels = [], 0

    # Target: chuyen doi mot mang cac nhan lop thanh mot dang ma hoa one-hot. Một ma trận one-hot với mỗi hàng biểu diễn một nhãn lớp,
    # trong đó chỉ có một phần tử bằng 1 và tất cả các phần tử khác bằng 0
    def convert_to_one_hot(y, C):
        return np.eye(C, dtype=np.uint8)[y.reshape(-1)]

    # vertex_tags la mot list cac nhan cac dinh trong do thi. Neu khac None thi chay qua cac vertex_tag va one_hot chung.
    if vertex_tags is not None:
        initial_feature_channels = tags_size
        print("X: one-hot vertex tag, tag size %d." % initial_feature_channels)
        for tag in vertex_tags:
            x = convert_to_one_hot(np.array(tag), initial_feature_channels)
            X.append(x)
    # Nguoc lai se chuan hoa roi them vao mang X.
    else:
        print("X: normalized node degree.")
        for graph in graphs_adj:
            degree_total = np.sum(graph, axis=1)
            X.append(np.divide(degree_total, np.sum(degree_total)).reshape(-1, 1))
        initial_feature_channels = 1
    X = np.array(X)
    #print(X)
    # doan code xay dung cac embedding features cho cac dinh trong do thi bang cach ket hop cac dac trung hien co voi cac dac trung nhung neu chung
    # co san.
    if embedding_feature is not None:
        print("embedding feature has considered")
        # build embedding for enclosing sub-graph
        sub_graph_emb = []
        for sub_nodes in sub_graphs_nodes:
            sub_graph_emb.append(embedding_feature[sub_nodes])
        for i in range(len(X)):
            #print(X[i].shape)
            #print(sub_graph_emb[i].shape)
            X[i] = np.concatenate([X[i], sub_graph_emb[i]], axis=1)
        # so luong kenh dac trung ban dau duoc cap nhat thanh so luong kenh dac trung dau tien trong X.
        initial_feature_channels = len(X[0][0])
    if explicit_feature is not None:
        initial_feature_channels = len(X[0][0])
        pass
    print("so, initial feature channels: ", initial_feature_channels)
    return np.array(D_inverse), graphs_adj, Y, X, node_size_list, initial_feature_channels  # ps, graph_adj is A_title