import networkx as nx
import numpy as np

def split_train_test(D_inverse, A_tilde, X, Y, nodes_size_list, rate=0.1):
    # xao tron du lieu truoc khi chia thanh 2 tap train va test
    print("split training and test data...")
    state = np.random.get_state()
    np.random.shuffle(D_inverse)
    np.random.set_state(state)
    np.random.shuffle(A_tilde)
    np.random.set_state(state)
    np.random.shuffle(X)
    np.random.set_state(state)
    np.random.shuffle(Y)
    np.random.set_state(state)
    np.random.shuffle(nodes_size_list)
    data_size = Y.shape[0]
    training_set_size, test_set_size = int(data_size * (1 - rate)), int(data_size * rate)
    D_inverse_train, D_inverse_test = D_inverse[: training_set_size], D_inverse[training_set_size:]
    A_tilde_train, A_tilde_test = A_tilde[: training_set_size], A_tilde[training_set_size:]
    X_train, X_test = X[: training_set_size], X[training_set_size:]
    Y_train, Y_test = Y[: training_set_size], Y[training_set_size:]
    nodes_size_list_train, nodes_size_list_test = nodes_size_list[: training_set_size], nodes_size_list[training_set_size:]
    print("about train: positive examples(%d): %s, negative examples: %s."
          % (training_set_size, np.sum(Y_train == 1), np.sum(Y_train == 0)))
    print("about test: positive examples(%d): %s, negative examples: %s."
          % (test_set_size, np.sum(Y_test == 1), np.sum(Y_test == 0)))
    return D_inverse_train, D_inverse_test, A_tilde_train, A_tilde_test, X_train, X_test, Y_train, Y_test, \
           nodes_size_list_train, nodes_size_list_test

# Ham dung de gan nhan cho cac nut trong do thi dua tren mot thuat toan goi la node labeling
def node_labeling(graph_adj, network_type):
    nodes_size = len(graph_adj)
    # Tao ra mot do thi ma tran ke
    G = nx.Graph(data=graph_adj) if network_type == 0 else nx.DiGraph(data=graph_adj)
    if len(G.nodes()) == 0:
        return [1, 1]
    tags = []
    # chay tu nut thu 2. Thuat toan tinh toan do dai duong di ngan nhat tu nut 0 va 1 cho den cac nut nay
    for node in range(2, nodes_size):
        try:
            dx = nx.shortest_path_length(G, 0, node)
            dy = nx.shortest_path_length(G, 1, node)
        except nx.NetworkXNoPath:
            tags.append(0)
            continue
        d = dx + dy
        div, mod = np.divmod(d, 2)
        tag = 1 + np.min([dx, dy]) + div * (div + mod - 1)
        tags.append(tag)
    return [1, 1] + tags