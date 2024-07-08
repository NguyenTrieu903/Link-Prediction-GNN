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

    # KHỞI TẠO VÀ CHUẨN BỊ DỮ LIỆU
    """Chia dữ liệu thành tập huấn luyện và tập kiểm tra dựa trên test_ratio"""
    test_size = int(len(positive) * test_ratio)
    train_pos, test_pos = positive[:-test_size], positive[-test_size:]
    train_neg, test_neg = negative[:-test_size], negative[-test_size:]

    """
        Tạo ma trận kề A dựa trên các cạnh dương trong tập huấn luyện. 
        Nếu đồ thị là vô hướng (network_type == 0), các cạnh sẽ được thêm vào cả hai hướng.
    """
    A = np.zeros([nodes_size, nodes_size], dtype=np.uint8)
    A[train_pos[:, 0], train_pos[:, 1]] = 1
    if network_type == 0:
        A[train_pos[:, 1], train_pos[:, 0]] = 1

    # HÀM TÍNH AUC CỦA HAI PHƯƠNG PHÁP CN VÀ AA:
    def calculate_auc(scores, test_pos, test_neg):
        pos_scores = scores[test_pos[:, 0], test_pos[:, 1]] #Trích xuất điểm số cho các mẫu positive từ ma trận scores
        neg_scores = scores[test_neg[:, 0], test_neg[:, 1]] #Trích xuất điểm số cho các mẫu negative từ ma trận scores
        s = np.concatenate([pos_scores, neg_scores]) #Mảng s chứa tất cả các điểm số của cả positive và negative samples.
        y = np.concatenate([np.ones(len(test_pos), dtype=np.int8), np.zeros(len(test_neg), dtype=np.int8)]) #Mảng y chỉ định nhãn (label) tương ứng với mỗi điểm số trong s (1 cho positive và 0 cho negative).
        assert len(s) == len(y) # Kiểm tra độ dài của s và y và đảm bảo chúng có cùng độ dài.
        auc = metrics.roc_auc_score(y_true=y, y_score=s) #tính auc với y_true là mảng chứa nhãn thực tế (ground truth) từ y và y_score là mảng chứa điểm số dự đoán từ s.
        return auc

    # determine the h value
    # doan code dung de tinh toan do tuong dong giua cac nut trong do thi dua tren thong tin cau truc.
    # 2 do tuong dong duoc tinh o day la: cn(Common Neighbors) va aa(Adamic-Adar).... Sau do so sanh xe auc cua phuong phap nao lon hon thi su dung phuong phap do
    
    # XÁC ĐỊNH GIÁ TRỊ HOP NẾU NÓ LÀ AUTO
    if hop == "auto":

        # Common Neighbors (cn)
        # toán tích chập của ma trận A với chính nó và trả về kết quả.
        def cn():
            return np.matmul(A, A) #hàm matmul chỉ xử lý đối với đối tượng là ma trận

        # Adamic-Adar (aa)
        # tính toán một phép nhân ma trận giữa A và A_ và trả về kết quả.
        def aa():
            A_ = A / np.log(A.sum(axis=1)) #  chia ma trận A cho logarit tổng của mỗi hàng của A
            A_[np.isnan(A_)] = 0 #thiết lập các phần tử NaN thành 0
            A_[np.isinf(A_)] = 0 # thiết lập các phần tử vô cùng thành 0
            return A.dot(A_) # thực hiện phép nhân ma trận giữa A và A_ -> hàm dot thì xử lý cả đối tượng là ma trận hoặc mảng.

        cn_scores, aa_scores = cn(), aa()
        cn_auc = calculate_auc(cn_scores, test_pos, test_neg) 
        aa_auc = calculate_auc(aa_scores, test_pos, test_neg)

        """
            - Ở đây giá trị tham số hop là 1 thì có nghĩa trích xuất tiểu với độ rộng là các đỉnh nối trực tiếp với đỉnh mục tiêu.
            Còn với 2 là ngoài các đỉnh láng giềng trực tiếp, các đỉnh láng giềng của các đỉnh láng giềng đó cũng được thêm vào tiểu đồ thị.
            - Việc sử dụng 2 pp này nhằm tính toán hệ số tương đồng giữa các nút trong đồ thị dựa trên thông tin cấu trúc từ đó tạo ra các tiểu
            đồ thị có tính tương đồng nhau. Còn việc mở rộng giá trị của tham số "hop" có thể không mang lại cải thiện đáng kể, trong khi lại làm 
            tăng độ phức tạp của hệ thống.
        """
        if cn_auc > aa_auc:
            print("cn(first order heuristic): %f > aa(second order heuristic) %f." % (cn_auc, aa_auc))
            hop = 1
        else:
            print("aa(second order heuristic): %f > cn(first order heuristic) %f. " % (aa_auc, cn_auc))
            hop = 2

    print("hop = %s." % hop)

    # TẠO ĐỒ THỊ G VÀ TRÍCH XUẤT TIỂU ĐỒ THỊ
    G = nx.Graph() if network_type == 0 else nx.DiGraph() #tạo đồ thị G
    G.add_nodes_from(set(positive[:, 0]) | set(positive[:, 1]) | set(negative[:, 0]) | set(negative[:, 1])) #thêm các đỉnh vào đồ thị G
    # G.add_nodes_from(set(sum(positive.tolist(), [])) | set(sum(negative.tolist(), [])))
    G.add_edges_from(train_pos) #thêm các cạnh vào đồ thị

    # TRÍCH XUẤT TIỂU ĐỒ THỊ
    """
        các đồ thị con (graphs_adj)
        nhãn của các cạnh (labels)
        nhãn của các đỉnh (vertex_tags)
        kích thước của các đồ thị con (node_size_list)
        danh sách các đỉnh trong các đồ thị con (sub_graphs_nodes)
        kích thước của tập hợp các nhãn (tags_size).
    """
    graphs_adj, labels, vertex_tags, node_size_list, sub_graphs_nodes = [], [], [], [], []
    for graph_label, data in enumerate([negative, positive]): #dùng vòng lặp enumerate để duyệt qua từng tập dữ liệu (negative và positive). 
        print("for %s. " % "negative" if graph_label == 0 else "positive")
        for node_pair in tqdm(data): #duyệt qua mỗi cặp đỉnh (node_pair) trong tập dữ liệu đó.
            sub_nodes, sub_adj, vertex_tag = extract_subgraph(node_pair, G, A, hop, network_type, max_neighbors) #ta sử dụng hàm extract_subgraph để trích xuất thông tin về đồ thị con xung quanh cặp đỉnh đó
            graphs_adj.append(sub_adj) #Thêm ma trận kề sub_adj vào danh sách graphs_adj.
            vertex_tags.append(vertex_tag) #Thêm nhãn đỉnh vertex_tag vào danh sách vertex_tags.
            node_size_list.append(len(vertex_tag)) #Thêm số lượng đỉnh của đồ thị con len(vertex_tag) vào danh sách node_size_list.
            sub_graphs_nodes.append(sub_nodes) #Thêm đồ thị con sub_nodes vào danh sách sub_graphs_nodes.
    assert len(graphs_adj) == len(vertex_tags) == len(node_size_list)
    # dung de tao ma tran nhan cho cac canh trong do thi. tao thanh ma tran co 1 cot va so hang la tong so cac canh trong do thi.
    # Ket qua la mot ma tran nhan, trong do moi hang dai dien cho 1 canh trong do thi va mot nhan tuong ung.
    labels = np.concatenate([np.zeros(len(negative), dtype=np.uint8), np.ones(len(positive), dtype=np.uint8)]).reshape(
        -1, 1)

    # CHUẨN HÓA NHÃN CÁC NODE:
    vertex_tags_set = set() #Khởi tạo một tập hợp rỗng để chứa các nhãn duy nhất.
    for tags in vertex_tags:
        vertex_tags_set = vertex_tags_set.union(set(tags)) #Tạo tập hợp các nhãn duy nhất
    vertex_tags_set = list(vertex_tags_set) #Chuyển tập hợp nhãn thành danh sách để sử dụng sau này.
    tags_size = len(vertex_tags_set) #Xác định số lượng nhãn duy nhất
    print("tight the vertices tags.")
    # kiem tra xem tat ca cac phan tu trong vertex_tags_set co tao thanh mot chuoi so nguyen lien tuc tu 0 den len(vertex_tags_set-1) hay khong.
    # Dieu nay dung de dam bao tinh day du va dung dan cua cac nhan duoc gan cho cac nut trong do thi.
    if set(range(len(vertex_tags_set))) != set(vertex_tags_set): #Kiểm tra tính liên tục của các nhãn đỉnh
        vertex_map = dict([(x, vertex_tags_set.index(x)) for x in vertex_tags_set]) #Chuẩn hóa các nhãn nếu chúng không liên tục
        for index, graph_tag in tqdm(enumerate(vertex_tags)): #Ánh xạ lại các nhãn trong vertex_tags
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
    # Khởi tạo tập hợp các đỉnh trong tiểu đồ thị
    sub_graph_nodes = set(node_pair) #ban đầu chứa hai đỉnh trong node_pair
    nodes = list(node_pair) # là danh sách chứa các đỉnh trong node_pair.
    nodes = list(nodes)
    # Mở rộng tiểu đồ thị:
    for i in range(int(hop)):
        np.random.shuffle(nodes) #Trong mỗi bước lặp, các đỉnh trong nodes được xáo trộn ngẫu nhiên.
        for node in nodes:
            neighbors = list(nx.neighbors(G, node)) #Lấy danh sách neighbors chứa các đỉnh láng giềng của node từ đồ thị G
            if len(sub_graph_nodes) + len(neighbors) < max_neighbors: #số lượng đỉnh trong sub_graph_nodes và neighbors nhỏ hơn max_neighbors
                sub_graph_nodes = sub_graph_nodes.union(neighbors) #thêm tất cả các đỉnh láng giềng vào sub_graph_nodes
            else: 
                np.random.shuffle(neighbors)
                sub_graph_nodes = sub_graph_nodes.union(neighbors[:max_neighbors - len(sub_graph_nodes)]) #thêm một phần số lượng đỉnh láng giềng ngẫu nhiên vào sub_graph_nodes, sao cho tổng số lượng đỉnh trong sub_graph_nodes không vượt quá max_neighbors
                break
        nodes = sub_graph_nodes - set(nodes) #Cập nhật danh sách nodes với các đỉnh mới được thêm vào sub_graph_nodes
    sub_graph_nodes.remove(node_pair[0]) #Loại bỏ đỉnh node_pair[0] khỏi sub_graph_nodes.
    if node_pair[0] != node_pair[1]: #Nếu node_pair[0] khác node_pair[1], loại bỏ đỉnh node_pair[1] khỏi sub_graph_nodes.
        sub_graph_nodes.remove(node_pair[1])
    sub_graph_nodes = [node_pair[0], node_pair[1]] + list(sub_graph_nodes)
    sub_graph_adj = A[sub_graph_nodes, :][:, sub_graph_nodes] #Trích xuất ma trận kề sub_graph_adj từ ma trận kề ban đầu A dựa trên sub_graph_nodes.
    sub_graph_adj[0][1] = sub_graph_adj[1][0] = 0 #Đặt giá trị 0 cho một phần tử trong ma trận kề sub_graph_adj để đảm bảo rằng cặp đỉnh node_pair không có kết nối trực tiếp.

    # labeling(coloring/tagging)
    vertex_tag = utils.node_labeling(sub_graph_adj, network_type) #gán nhãn cho các đỉnh trong tiểu đồ thị
    return sub_graph_nodes, sub_graph_adj, vertex_tag