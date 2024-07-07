import torch
from torch import Tensor
from torch_scatter import scatter_add

# Hàm này có chức năng tính toán bậc (degree) của các đỉnh trong đồ thị, dựa trên các chỉ số cạnh ei và số lượng đỉnh num_node
@torch.jit.script
def degree(ei: Tensor, num_node: int):
    return scatter_add(torch.ones_like(ei[1]), ei[1], dim_size=num_node)

# Hàm này thực hiện phép nhân "set multiplication" giữa hai tensor a và b
@torch.jit.script
def set_mul(a: Tensor, b: Tensor):
    # Biến đổi tensor a thành một tensor có kích thước mới là (số phần tử của a, 1).
    # Điều này đảm bảo rằng mỗi phần tử của a sẽ được biểu diễn riêng lẻ trong một hàng.
    a = a.reshape(-1, 1)
    # Biến đổi tensor b thành một tensor có kích thước mới là (1, số phần tử của b).
    # Điều này đảm bảo rằng mỗi phần tử của b sẽ được biểu diễn riêng lẻ trong một cột.
    b = b.reshape(1, -1)
    # Mở rộng tensor a sao cho số lượng hàng bằng với ban đầu và
    # số lượng cột bằng với số lượng cột của b. Các phần tử trong mỗi hàng sẽ giống nhau theo cột.
    a = a.expand(-1, b.shape[1])
    # Mở rộng tensor b sao cho số lượng cột bằng với ban đầu và số lượng hàng bằng với số lượng hàng của a.
    # Các phần tử trong mỗi cột sẽ giống nhau theo hàng.
    b = b.expand(a.shape[0], -1)
    # Nối hai tensor đã được biến đổi theo cột (mỗi phần tử của a với mỗi phần tử của b) thành một tensor mới.
    # Sau đó, reshape lại tensor này để có kết quả cuối cùng là một tensor có kích thước (số phần tử của a * số phần tử của b, 2),
    # trong đó cột đầu tiên chứa các phần tử của a, cột thứ hai chứa các phần tử của b.
    return torch.cat((a.reshape(-1, 1), b.reshape(-1, 1)), dim=-1)

@torch.jit.script
def get_ei2(n_node: int, pos_edge, pred_edge):
    """
        được sử dụng để chuẩn bị dữ liệu đầu vào bằng cách xây dựng các tensor chứa các chỉ số cạnh và các đặc trưng tương ứng để huấn
        luyện mô hình.
        Sử dụng để biến đổi các cạnh thành các chỉ số có thể được sử dụng để huấn luyện và dự đoán.
    """
    # Nối hai tensor pos_edge và pred_edge theo chiều cuối cùng (dim=-1).
    # Đây là tập hợp các cạnh bao gồm cả các cạnh thực tế (pos_edge) và các cạnh dự đoán (pred_edge).
    edge = torch.cat((pos_edge, pred_edge), dim=-1)  # pos.transpose(0, 1)
    # Tạo một tensor chứa các chỉ số từ 0 đến số lượng cạnh trong edge
    idx = torch.arange(edge.shape[1], device=edge.device)
    # Tạo một tensor chứa các chỉ số từ 0 đến số lượng cạnh trong pos_edge
    idx_pos = torch.arange(pos_edge.shape[1], device=edge.device)
    # for qua từng node.
    # Chọn các chỉ số trong idx_pos mà có giá trị node đích bằng i. Chọn các chỉ số trong idx mà có giá trị node nguồn bằng i.
    # Sau đó áp dụng hàm set_mul (phép nhân 2 tensor) để tính chỉ số cạnh
    edge2 = [
        set_mul(idx_pos[pos_edge[1] == i], idx[edge[0] == i])
        for i in range(n_node)
    ]
    # Nối danh sách các tensor trong edge2 theo chiều 0 để tạo một tensor lớn.
    return torch.cat(edge2, dim=0).t()

@torch.jit.script
def check_in_set(target, set):
    """
        Hàm này hoạt động bằng cách so sánh các phần tử của target với các phần tử của set và trả về một tensor với giá trị True tại các vị trí mà
        phần tử tương ứng của target có mặt trong set.
    """
    """
    a là tensor target được định hình lại thành một cột (shape: (n, 1)).
    b là tensor set được định hình lại thành một hàng (shape: (1, m)).
    """
    a = target.reshape(-1, 1)
    b = set.reshape(1, -1)
    out = []
    """
        cutshape xác định kích thước của các khối nhỏ hơn để xử lý, nhằm tránh vượt quá giới hạn bộ nhớ.
    """
    cutshape = 1024 * 1024 * 1024 // b.shape[1]
    """
        Vòng lặp này chia a thành các khối nhỏ với kích thước cutshape để so sánh với b.
        a[i:i + cutshape] == b tạo ra một tensor boolean nơi mỗi phần tử trong khối a được so sánh với mỗi phần tử trong b.
        torch.sum(..., dim=-1) tính tổng số các giá trị True dọc theo hàng (dim -1), kết quả là một tensor 
                                với số lần xuất hiện của các phần tử target trong set.
        torch.cat([...]) nối tất cả các kết quả lại với nhau thành một tensor duy nhất.
    """
    out = torch.cat([
        torch.sum((a[i:i + cutshape] == b), dim=-1)
        for i in range(0, a.shape[0], cutshape)
    ])
    """
        out chứa số lần xuất hiện của mỗi phần tử trong target trong set
    """
    return out

@torch.jit.script
def blockei2(ei2, blocked_idx):
    """
        Hàm này dùng để bỏ các cạnh mà chúng đã xuất hiện trong blocked_idx
        Hoạt động:
            Gọi hàm check_in_set để trả về một tensor tại các vị trí mà phần tử tương ứng nằm trong blocked_idx.
            Sau đó đảo ngược giá trị boolean để có một tensor với giá trị True tại các vị trí mà phần tử tương ứng không nằm trong blocked_idx.
            Ex:
                ei2 = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]])
                blocked_idx = torch.tensor([1, 3])

                filtered_ei2 = blockei2(ei2, blocked_idx)
                print(filtered_ei2) => tensor([[0, 2],
                                                [4, 6]])
        Lọc các cạnh bị chặn: Loại bỏ các cạnh mà nguồn của chúng nằm trong danh sách bị chặn (blocked_idx). Điều này có thể hữu ích khi muốn
                        loại trừ một số cạnh nhất định khỏi quá trình huấn luyện hoặc đánh giá.
        Xử lý dữ liệu đồ thị: Dễ dàng chọn lọc và xử lý các cạnh dựa trên điều kiện cụ thể, giúp giảm nhiễu và tập trung vào các cạnh có liên quan.
    """
    return ei2[:, torch.logical_not(check_in_set(ei2[0], blocked_idx))]


@torch.jit.script
def idx2mask(num: int, idx):
    """
        Dùng để tạo một mask tensor từ một danh sách các chỉ số
        Ex:
            num = 10
            idx = torch.tensor([1, 3, 5, 7])
            => tensor([False,  True, False,  True, False,  True, False,  True, False, False])
        Trong bài toán dự đoán liên kết, hàm idx2mask có thể được sử dụng để:
            Tạo các mặt nạ (masks) cho tập dữ liệu: Đánh dấu các cạnh (edges) hoặc các node trong đồ thị mà bạn muốn chú ý hoặc bỏ qua trong các
            bước tính toán tiếp theo.
            Lọc các phần tử: Sử dụng mặt nạ để chọn lọc các phần tử từ tensor khác (chẳng hạn như các đặc trưng của node hoặc cạnh) dựa trên chỉ
            số được cung cấp.
        => Giúp dễ dàng chọn lọc và xử lý các phần tử trong tensor khác dựa trên chỉ số.
    """
    mask = torch.zeros((num), device=idx.device, dtype=torch.bool)
    mask[idx] = True
    return mask


# @torch.jit.script
def sample_block(sample_idx, size, ei, ei2=None):
    """
        Hàm này dùng để lấy các sample chưa được thực thi trong sample_idx
    """
    """
        ea (edge attributes): 
        ea = torch.ones((ei.shape[-1],), dtype=torch.float, device=ei.device) => dùng để khởi tạo một tensor ea chứa các giá trị là 1, có kích thước 
        bằng với số lượng cạnh trong ei 
    """
    ea = torch.ones((ei.shape[-1],), dtype=torch.float, device=ei.device)
    """
         Sử dụng để tạo ra một tensor mới ea_new, loại bỏ các phần tử có chỉ số trong sample_idx khỏi tensor ea
         Ex:
            ea = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
            sample_idx = torch.tensor([1, 3])
            
            ea_new = ea[torch.logical_not(idx2mask(ea.shape[0], sample_idx))]
            print(ea_new) => tensor([1., 3., 5.])
    """
    ea_new = ea[torch.logical_not(idx2mask(ei.shape[1], sample_idx))]
    """
        Tương tự code trên tạo ra một tensor mới ei_new, loại bỏ các cạnh có chỉ số trong sample_idx khỏi tensor ei
        Ex:
            ei = torch.tensor([[0, 1, 2], [3, 4, 5]])  # Một tensor đại diện cho các cạnh
            sample_idx = torch.tensor([1, 3])  # Các chỉ số của các cạnh bạn muốn loại bỏ
            
            ei_new = ei[:, torch.logical_not(idx2mask(ei.shape[1], sample_idx))]
            print(ei_new) => tensor([[0, 2],
                                    [3, 5]])
    """
    ei_new = ei[:, torch.logical_not(idx2mask(ei.shape[1], sample_idx))]
    """
        Gọi hàm blockei2 để xử lý các cạnh đã xuất hiện trong sample_idx
    """
    ei2_new = blockei2(ei2, sample_idx) if ei2 is not None else None
    """
        Sử dụng để tạo một ma trận đồ thị thưa thớt (sparse adjacency matrix) từ ei_new và ea_new
        ei_new: Đây là một tensor chứa thông tin về các cạnh của đồ thị. Thường thì ei_new có dạng (2, num_edges) hoặc (num_edges, 2), trong đó 
            hàng đầu tiên (hoặc cột đầu tiên) chứa chỉ số của đỉnh nguồn của các cạnh và hàng thứ hai (hoặc cột thứ hai) chứa chỉ số của đỉnh đích của 
            các cạnh.
        ea_new: Đây là một tensor chứa thông tin về trọng số hoặc thuộc tính của các cạnh tương ứng với ei_new. ea_new có cùng số lượng phần tử với 
            ei_new và thường có dạng (num_edges,) hoặc (1, num_edges).
        (size, size): Đây là kích thước của ma trận đồ thị cuối cùng mà bạn muốn tạo ra. Kích thước này thường là số đỉnh của đồ thị. size là số 
            lượng đỉnh của đồ thị.
        Lợi ích và ứng dụng:
            Tiết kiệm bộ nhớ: Ma trận thưa thớt chỉ lưu trữ các giá trị khác không (non-zero values), giúp tiết kiệm bộ nhớ so với ma trận đầy đủ.
            Hiệu quả tính toán: Phù hợp cho các đồ thị lớn với số lượng cạnh lớn, vì các phép toán trên ma trận thưa thớt thường nhanh hơn so với ma 
                trận đầy đủ.
            Dễ dàng sử dụng: Ma trận thưa thớt có thể được truyền vào các thuật toán và hàm xử lý dữ liệu đồ thị của PyTorch một cách thuận tiện và 
                hiệu quả.
    """
    adj = torch.sparse_coo_tensor(ei_new, ea_new, (size, size))
    """
        Sử dụng để tính toán tổng các phần tử trên mỗi hàng của ma trận thưa thớt adj, sau đó chuyển đổi kết quả thành tensor mật độ (dense tensor), 
            rồi định hình lại thành một tensor 1 chiều.
        Tổng của các phần tử trên mỗi hàng của ma trận thưa thớt có thể cung cấp thông tin về bậc (degree) của các đỉnh trong đồ thị.
    """
    x_new = torch.sparse.sum(adj, dim=1).to_dense().to(torch.int64).reshape(-1)
    return ei_new, x_new, ei2_new


def reverse(edge_index):
    """
        Hàm này giúp tăng tính đa dạng cho dữ liệu huấn luyện.
        Bằng cách thay đổi giá trị của các nút dựa trên tính chẵn lẻ,
        ta có thể tạo ra các biến thể của cạnh và giúp mô hình học được nhiều đặc điểm hơn từ dữ liệu.
        Hoặc có thể sử dụng để điều chỉnh giá trị các nút để tránh xung đột hoặc trùng lặp trong quá trình xử lý đồ thị. Điều này có thể giúp đảm bảo
        rằng các giá trị nút là duy nhất hoặc tuân theo một số quy tắc nhất định.
    """
    """
        edge được tạo bằng cách điều chỉnh giá trị của edge_index[0] với tem0
        Điều này có nghĩa là nếu edge_index[0] là số lẻ, nó sẽ bị giảm đi 1, và nếu là số chẵn, nó sẽ tăng thêm 1
        edge_r được tạo bằng cách tương tự với tem1
    """
    tem0 = 1 - (edge_index[0] > edge_index[0] // 2 * 2).to(torch.long) * 2
    tem1 = 1 - (edge_index[1] > edge_index[1] // 2 * 2).to(torch.long) * 2
    edge = torch.cat([(edge_index[0] + tem0).unsqueeze(0), edge_index[1].unsqueeze(0)])
    edge_r = torch.cat([edge_index[0].unsqueeze(0), (edge_index[1] + tem1).unsqueeze(0)])
    # return edge_index
    return edge, edge_r


import math

"""
    Hàm này được sử dụng để biến đổi và mở rộng dữ liệu về edges của graph.
        Khi for_index = False -> Kết quả là một tensor mà các cạnh được "nhân đôi" để bao gồm cả hướng ngược lại (đối xứng), 
        giúp mô hình học được các mối quan hệ đối xứng trong đồ thị.
        Khi for_index = True -> xử lý các chỉ số của cạnh (edge indices) để tạo ra các chỉ số mới với giá trị nhân đôi và cộng thêm 1, 
    for_index = False 
            x before tensor([[1, 2, 3, 4],
                [5, 6, 8, 2]])
            x for for_index is False  tensor([[1, 5, 2, 6, 3, 8, 4, 2],
                                            [5, 1, 6, 2, 8, 3, 2, 4]])
            tensor([[1, 5, 2, 6, 3, 8, 4, 2],
                    [5, 1, 6, 2, 8, 3, 2, 4]])
    for_index = True 
            x before tensor([1, 2, 3, 4])
            x for for_index is True  tensor([2, 3, 4, 5, 6, 7, 8, 9])
            tensor([2, 3, 4, 5, 6, 7, 8, 9])
    Việc này giúp tăng cường dữ liệu, tạo thêm nhiều mẫu từ cùng 1 tập dữ liệu, giúp mô hình học được từ nhiều góc độ khác nhau. 
    Giúp mô hình tách biệt các node hoặc các mẫu dựa trên một số tính chất hoặc điều kiện cụ thể 
"""
def double(x, for_index=False):
    if not for_index:
        row, col = x[0].reshape(1, x.shape[1]), x[1].reshape(1, x.shape[1])
        x = torch.cat([row, col, col, row], 0).t()
        x = x.reshape(-1, 2).t()
    else:
        x = x.reshape(1, x.shape[0])
        x = torch.cat([2 * x, 2 * x + 1], 0).t()
        x = x.reshape(-1, 1).t().squeeze()
    return x


# used to random split edges
def random_split_edges(data, val_ratio: float = 0.05,
                       test_ratio: float = 0.1):
    # get attributes from data such as: num_nodes, row, col....
    num_nodes = data.num_nodes
    row, col = data.edge_index
    edge_attr = data.edge_attr
    data.edge_index = data.edge_attr = None
    # mask not necessary
    # Return upper triangular portion.
    # mask = row < col
    # print("mask", mask)
    # row, col = row[mask], col[mask]
    # print("row after", row)
    # print("col after", col)
    # if edge_attr is not None:
    #     edge_attr = edge_attr[mask]

    # get number node of validation and test
    n_v = int(math.floor(val_ratio * row.size(0)))
    n_t = int(math.floor(test_ratio * row.size(0)))

    # handle Positive edges.

    # Returns a random permutation of integers from 0 to row.size(0) - 1
    perm = torch.randperm(row.size(0))
    row, col = row[perm], col[perm]
    if edge_attr is not None:
        edge_attr = edge_attr[perm]

    r, c = row[:n_v], col[:n_v]
    data.val_pos_edge_index = torch.stack([r, c], dim=0)
    if edge_attr is not None:
        data.val_pos_edge_attr = edge_attr[:n_v]

    r, c = row[n_v:n_v + n_t], col[n_v:n_v + n_t]
    data.test_pos_edge_index = torch.stack([r, c], dim=0)
    if edge_attr is not None:
        data.test_pos_edge_attr = edge_attr[n_v:n_v + n_t]

    r, c = row[n_v + n_t:], col[n_v + n_t:]
    data.train_pos_edge_index = torch.stack([r, c], dim=0)

    # Negative edges.
    # Tạo ra các một ma trận 2-D với giá trị là 1 và có số cột bằng num_nodes và số hàng bằng num_nodes
    neg_adj_mask = torch.ones(num_nodes, num_nodes, dtype=torch.uint8)
    # Sau khi tạo xong, dùng hàm triu() để lấy tất cả các phần tử nằm trên hoặc trên đường chéo chính của ma trận. diagonal=1, trả về các phần tử nằm
    # phía trên đường chéo chính của ma trận (từ đường chéo chính đi lên chứ khoogn phải nằm trên đường chéo chính )
    neg_adj_mask = neg_adj_mask.triu(diagonal=1)
    # Sau đó gán các giá trị có row và col trong ma trận này là 0. Nghĩa là chỗ nào số 0 là chỗ đó có liên kết
    neg_adj_mask[row, col] = 0

    # Dùng hàm nonzero để lấy ra các giá trị khác 0. Nghĩa là các cnahj này chưa được liên kết trước đó => negative edge
    # Tham số as_tuple=False đảm bảo rằng kết quả trả về là một tensor có hai cột,
    # mỗi hàng của nó biểu thị một chỉ số của một phần tử khác không trong neg_adj_mask.
    # Vì thế muốn lấy chỉ số các row và col thì phải dùng t() để hoán đổi chiều của tensor.
    neg_row, neg_col = neg_adj_mask.nonzero(as_tuple=False).t()
    # Sau đó dùng hàm random để random lấy giá trị sau mỗi lần split.
    # Chỉ lấy từ 0 đến n_v + n_t là muốn neg cho train không vượt quá tỉ lệ đưa ra
    perm = torch.randperm(neg_row.size(0))
    neg_row, neg_col = neg_row[perm], neg_col[perm]

    # Sau khi xác định các cạnh neg thì cần dánh dấu lại các cạnh này giả định đã liên kết
    neg_adj_mask[neg_row, neg_col] = 0
    data.train_neg_adj_mask = neg_adj_mask

    row, col = neg_row[:n_v], neg_col[:n_v]
    data.val_neg_edge_index = torch.stack([row, col], dim=0)

    row, col = neg_row[n_v:n_v + n_t], neg_col[n_v:n_v + n_t]
    data.test_neg_edge_index = torch.stack([row, col], dim=0)

    return data
