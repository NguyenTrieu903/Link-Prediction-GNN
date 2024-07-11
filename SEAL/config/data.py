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
    # ĐỌC DỮ LIỆU TỪ FILE
    positive_df = pd.read_csv('/home/nhattrieu-machine/Documents/2WL_link_pred-main/raw_data/fb-pages-food/fb-pages-food.edges', delimiter=',', dtype=int)
    # chuyển dữ liệu được đọc được thành mảng Numpy
    positive = positive_df.to_numpy()

    # VẼ ĐỒ THỊ G VỚI CÁC CẠNH HIỆN CÓ TRONG FILE
    G = nx.Graph() if network_type == 0 else nx.DiGraph() # tạo biến đồ thị G
    G.add_edges_from(positive) # add các cạnh hiện có vào biến đồ thị G

    # TẠO DANH SÁCH CÁC CẠNH KHÔNG TỒN TẠI TRONG ĐỒ THỊ G
    negative_all = list(nx.non_edges(G)) #tạo danh sách chứa tất cả các cạnh không tồn tại trong G
    np.random.shuffle(negative_all) #xáo trộn ngẫu nhiên thứ tự các phần tử trong danh sách các cạnh không tồn tại trong G
    negative = np.asarray(negative_all[:len(positive)]) #Hàm này chọn một số lượng cạnh không tồn tại bằng với số lượng cạnh tồn tại trong đồ thị G.
    np.random.shuffle(positive) #xáo trộn ngẫu nhiên thứ tự các phần tử trong danh sách các cạnh tồn tại trong G
    """ việc xáo trộn các cạnh trong 2 danh sách giúp đảm bảo tính ngẫu nhiên trong quá trình huấn luyện."""

    print("positve examples: %d, negative examples: %d." % (len(positive), len(negative)))

    # ĐIỀU CHỈNH GIÁ TRỊ CÁC CẠNH TRONG DANH SÁCH POSITIVE VÀ NEGATIVE SAO CHO CÁC GIÁ TRỊ NÀY BẮT ĐẦU TỪ 0 THAY VÌ 1
    """ 
        Vì các thuật toán xử lý đồ thị thường bắt đầu đánh số các node (đỉnh) từ 0. 
        Nếu dữ liệu đầu vào sử dụng đánh số từ 1, bạn cần điều chỉnh lại để đảm bảo tính nhất quán 
        trong quá trình xử lý và huấn luyện mô hình. Nếu không có bước này và dữ liệu đầu vào bắt đầu từ 1, 
        có thể dẫn đến lỗi hoặc kết quả không chính xác trong quá trình xử lý và huấn luyện mô hình. 
    """
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

    # TẠO DỮ LIỆU ĐÀO TẠO CHO QUÁ TRÌNH NHÚNG
    test_size = int(test_ratio * positive.shape[0])
    train_posi, train_nega = positive[:-test_size], negative[:-test_size]

    # TẠO ĐỒ THỊ A CÓ HƯỚNG HOẶC KHÔNG DỰA VÀO BIẾN "network_type"
    A = nx.Graph() if network_type == 0 else nx.DiGraph()

    # THÊM CÁC CẠNH VÀO ĐỒ THỊ A
    """
        - Đầu tiên, ta thêm các cạnh positive vào trong đồ thị A và gán với trọng số là 1;
        - Tiếp theo, nếu biến negative_injection có giá trị true thì ta cũng thêm các cạnh trong negative vào đồ thị A
        - Ở đây, vì để tăng khả năng học được cấu trúc, mối quan hệ trong mạng và tăng khả năng dự đoán, khả năng tổng 
        quát hóa trên dữ liệu thì ta tiến hành cho phép thêm các cạnh trong nagative vào đồ thị A và cũng gán với trọng số là 1.
    """
    A.add_weighted_edges_from(
        np.concatenate([train_posi, np.ones(shape=[train_posi.shape[0], 1], dtype=np.int8)], axis=1))
    if negative_injection:
        A.add_weighted_edges_from(
            np.concatenate([train_nega, np.ones(shape=[train_nega.shape[0], 1], dtype=np.int8)], axis=1))
    line_graph = nx.line_graph(A) # đoạn này không có dùng
    
    # QUÁ TRÌNH EMBEĐING DATA
    """
        Ta khởi tạo một đồ thị “G” vô hướng từ đồ thị “A” sử dụng thuật toán Node2Vec, 
        với việc không có sự ưu tiên nào giữa các đỉnh cùng loại (gần nhau) hoặc khác loại (xa nhau) 
        thông qua tham số p = 1 (pronounced “p-factor”) và q = 1 (pronounced “q-factor”).
    """
    G = node2vec.Graph(A, is_directed=False if network_type == 0 else True, p=1, q=1)

    """tiền xử lý các xác suất chuyển đổi nhằm hướng dẫn các random walks (các bước di chuyển ngẫu nhiên trên đồ thị)."""
    G.preprocess_transition_probs()

    """
        Hàm này tạo ra các random walk trong đồ thị G. num_walks là số lượng random walk được tạo ra cho mỗi đỉnh trong đồ thị, 
        và walk_length là độ dài của mỗi random walk. Sau đó,ta chuyển đổi các đỉnh trong mỗi random walk từ dạng số nguyên sang dạng chuỗi.
    """
    walks = G.simulate_walks(num_walks=10, walk_length=80)
    walks = [list(map(str, walk)) for walk in walks]

    """
        Tạo ra một mô hình Word2Vec từ danh sách các random walk vừa tạo bên trên. Trong đó:
        - walks: danh sách các random walk vừa tạo bên trên.
        - size: kích thước của vecto nhúng cho mỗi đỉnh trong đồ thị
        - window: xác định số lượng node xung quanh một node được xem xét trong quá trình huấn luyện.
        - min_count: quy định số lần xuất hiện tối thiểu của một node trong dữ liệu để được xem xét trong quá trình huấn luyện. 
        Một node có tần suất xuất hiện thấp hơn min_count sẽ bị bỏ qua và không được sử dụng để cập nhật các vector nhúng.
        - sg: Phương pháp huấn luyện: sg=1 cho Skip-gram
        - workers: Số lượng luồng được sử dụng trong quá trình huấn luyện.
        - iter: Số lần lặp lại quá trình huấn luyện trên dữ liệu.
    """
    model = Word2Vec(walks, size=dimension, window=10, min_count=0, sg=1, workers=8, iter=1)

    """
        Hàm này truy cập vào thuộc tính wv của mô hình Word2Vec để lấy vector nhúng (embedding) của các đỉnh trong đồ thị. 
        wv là một đối tượng chứa các phương thức và thuộc tính liên quan đến vector nhúng.
    """
    wv = model.wv

    # CHUẨN BỊ MA TRẬN NHÚNG
    embedding_feature, empty_indices, avg_feature = np.zeros([network_size, dimension]), [], 0
    for i in range(network_size):
        if str(i) in wv:
            embedding_feature[i] = wv.word_vec(str(i)) # Nếu node có nhúng (tồn tại trong wv), thêm nhúng vào ma trận embedding_feature.
            avg_feature += wv.word_vec(str(i)) # Nếu node có nhúng (tồn tại trong wv), tính tổng các vector nhúng của các đối tượng có sẵn.
        else:
            empty_indices.append(i) # Nếu node không có nhúng (không tồn tại trong wv), lưu trữ chỉ số node này vào empty_indices.
    # Tính giá trị trung bình của tất cả các nhúng hiện có và gán giá trị này cho các node không có nhúng (empty_indices).
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


