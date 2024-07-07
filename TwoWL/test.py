import torch
from torch import Tensor
def neg_mask():
    # Giả sử neg_row và neg_col có các giá trị như sau
    neg_row = torch.tensor([0, 1, 2, 3])
    neg_col = torch.tensor([0, 1, 2, 3])
    neg_adj_mask = torch.tensor([[0, 1, 2, 3, 4],
                                 [7, 8, 4, 3, 7],
                                 [45, 76, 4, 8, 6],
                                 [77, 44, 44, 33, 11],
                                 ])
    # Số lượng phần tử bạn muốn lấy từ hoán vị
    n_v = 2
    n_t = 1

    # Tạo một hoán vị ngẫu nhiên của chỉ số từ 0 đến neg_row.size(0) - 1
    perm = torch.randperm(neg_row.size(0))[:n_v + n_t]
    print(perm)
    # Áp dụng hoán vị ngẫu nhiên để lấy chỉ số
    neg_row, neg_col = neg_row[perm], neg_col[perm]
    neg_adj_mask[neg_row, neg_col] = 0
    print(neg_adj_mask)
    print("neg_row sau khi xáo trộn và lấy mẫu:", neg_row)
    print("neg_col sau khi xáo trộn và lấy mẫu:", neg_col)
def torch_cat():
    import torch

    train_pos = torch.tensor([[0, 1], [3, 4]])
    val_pos = torch.tensor([[6, 7], [8, 9]])
    test_pos = torch.tensor([[10, 11], [12, 13]])

    # Kết hợp các tensor dọc theo chiều cuối cùng
    edge_pos = torch.cat((train_pos, val_pos), dim=0)
    edge_pos_c = torch.cat((train_pos, val_pos), dim=1)
    print(edge_pos)
    print(edge_pos_c)

def double(x, for_index=False):
    print("x before", x)
    if not for_index:
        row, col = x[0].reshape(1, x.shape[1]), x[1].reshape(1, x.shape[1])
        x = torch.cat([row, col, col, row], 0).t()
        x = x.reshape(-1, 2).t()
        print("x for for_index is False ", x)
    else:
        x = x.reshape(1, x.shape[0])
        x = torch.cat([2 * x, 2 * x + 1], 0).t()
        x = x.reshape(-1, 1).t().squeeze()
        print("x for for_index is True ", x)
    return x
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
def get_ei2(n_node: int, pos_edge, pred_edge):
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
def reverse(edge_index):
    """
        Hàm này có chức năng để tạo ra các cạnh đối xứng (reverse edges) từ một tập hợp các cạnh ban đầu edge_index
    """
    a = edge_index[0] // 2 * 2
    b = edge_index[1] // 2 * 2
    tem0 = 1 - (edge_index[0] > edge_index[0] // 2 * 2).to(torch.long) * 2
    tem1 = 1 - (edge_index[1] > edge_index[1] // 2 * 2).to(torch.long) * 2
    edge = torch.cat([(edge_index[0] + tem0).unsqueeze(0), edge_index[1].unsqueeze(0)])
    edge_r = torch.cat([edge_index[0].unsqueeze(0), (edge_index[1] + tem1).unsqueeze(0)])
    # return edge_index
    return edge, edge_r

def mask ():
    x = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    # Tạo mask
    mask = torch.cat(
        [torch.ones([1, x.shape[0] // 2], dtype=bool),
         torch.zeros([1, x.shape[0] // 2], dtype=bool)]
    ).t().reshape(-1)

    # Lọc các phần tử của x dựa trên mask
    filtered_x = x[mask]
    filtered_x_ne = x[~mask]
    filtered_x = x[mask] * x[~mask]
    print(filtered_x)
if __name__ == '__main__':
    # res = get_ei2(5,
    #               torch.tensor([[0, 1, 2], [3, 4, 5]]),
    #               torch.tensor([[0, 3], [1, 2]])
    #               )
    # print(res)
    # edge_index = torch.tensor([[0, 1, 1, 2, 3, 4], [1,2,4,2,3,1]])
    # ed, edr = reverse(edge_index)
    # print("ed ",ed)
    # print("edr ", edr)
    # mask()
    # Tạo hai tensor ví dụ
    tensor1 = torch.tensor([1, 2, 3])
    tensor2 = torch.tensor([4, 5, 6])
    x_1, x_2 = torch.randn(2, 5), torch.randn(2, 5)
    print(x_1)
    print(x_2)
    # Xếp các tensor theo chiều 0
    stacked_tensor = torch.stack([tensor1, tensor2], dim=0)

    print(stacked_tensor.size())
    # Output: tensor([[1, 2, 3],
    #                 [4, 5, 6]])
    a = torch.arange(4.)
    print(torch.reshape((-1, 1)))