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
    #Tính toán kích thước của tập dữ liệu
    data_size = Y.shape[0]
    #Tính toán kích thước của tập train và test dựa trên tỷ lệ
    training_set_size, test_set_size = int(data_size * (1 - rate)), int(data_size * rate)
    #Chia dữ liệu thành các tập train và test dựa trên kích thước đã tính toán. 
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
    nodes_size = len(graph_adj) #tính số lượng đỉnh trong đồ thị
    # Tao ra mot do thi dua trên ma tran ke
    G = nx.Graph(data=graph_adj) if network_type == 0 else nx.DiGraph(data=graph_adj)
    if len(G.nodes()) == 0: #kiểm tra xem trong đồ thioj có đỉnh nào hay không
        return [1, 1]
    tags = [] #khởi tạo một danh sách rỗng để lưu trữ các nhãn (tags).
    # chay tu nut thu 2. Thuat toan tinh toan do dai duong di ngan nhat tu nut 0 va 1 cho den cac nut nay
    for node in range(2, nodes_size):
        try:
            dx = nx.shortest_path_length(G, 0, node) #tính toán độ dài đường đi ngắn nhất từ đỉnh 0 đến đỉnh node
            dy = nx.shortest_path_length(G, 1, node) #tính toán độ dài đường đi ngắn nhất từ đỉnh 1 đến đỉnh node
        except nx.NetworkXNoPath:
            tags.append(0) #Nếu không có đường đi giữa các đỉnh, một giá trị nhãn là 0 sẽ được thêm vào danh sách tags.
            continue
        d = dx + dy #Tính tổng độ dài đường đi từ đỉnh 0 đến node và từ đỉnh 1 đến node.
        div, mod = np.divmod(d, 2) #Tính phần nguyên (div) và phần dư (mod) khi chia tổng độ dài của đường đi cho 2
        tag = 1 + np.min([dx, dy]) + div * (div + mod - 1) #tính toán giá trị nhãn (tag) dựa trên độ dài đường đi dx, dy
        tags.append(tag) #Thêm giá trị nhãn (tag) vào danh sách tags.
    return [1, 1] + tags