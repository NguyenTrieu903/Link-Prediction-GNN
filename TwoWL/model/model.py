from torch import nn
from torch.nn.modules.dropout import Dropout
from torch_geometric.nn import GCNConv, GraphNorm

# from utils import reverse, sparse_bmm, sparse_cat, add_zero, edge_list
from TwoWL.utils import *

class LocalWLNet(nn.Module):
    def __init__(self,
                 max_x,
                 use_node_feat,
                 node_feat,
                 channels_1wl=32,
                 channels_2wl=24,
                 depth1=2,
                 depth2=1,
                 dp_lin0=0.2,
                 dp_lin1=0.30000000000000004,
                 dp_emb=0.4,
                 dp_1wl0=0.30000000000000004,
                 dp_2wl=0.30000000000000004,
                 dp_1wl1=0.1,
                 act0=True,
                 act1=False,
                 ):
        super().__init__()

        use_affine = False

        """
            Là một hàm lambda (hay hàm ẩn danh) được định nghĩa để tạo ra một chuỗi các lớp linear neural network (nn.Sequential).
             nn.Linear(a, b): 
                Chuyển đổi một tensor từ không gian đầu vào với kích thước a sang không gian đầu ra với kích thước b. 
                Mục đích biến đổi không gian đặc trưng của dữ liệu để phù hợp với mục tiêu của mô hình
             nn.LayerNorm(b, elementwise_affine=use_affine) if lnx else nn.Identity():  
                Nó chuẩn hóa các đặc trưng (features) đầu vào sao cho chúng có mean (trung bình) gần bằng 0 và độ lệch chuẩn gần bằng 1. 
                Tham số b là số lượng đặc trưng đầu ra của layer trước đó (sau khi đi qua nn.Linear).
                elementwise_affine=use_affine: Tham số này quyết định xem liệu layer chuẩn hóa có sử dụng tham số scale và shift để biến đổi dữ liệu 
                    sau chuẩn hóa hay không. Nếu use_affine là True, tức là sử dụng tham số; ngược lại là không sử dụng.
                nn.Identity(): Đây là một module trong PyTorch không làm thay đổi dữ liệu đầu vào mà chỉ trả về nó nguyên vẹn. Nó được sử dụng khi 
                    không cần áp dụng bất kỳ biến đổi nào lên dữ liệu.
            nn.Dropout(p=dp, inplace=True):
                Sử dụng để thực hiện phép dropout trong mạng nơ-ron. Phép dropout là một kỹ thuật regularization phổ biến trong quá trình huấn luyện 
                    mạng nơ-ron để ngăn chặn việc overfitting.
                Tham số p=dp xác định xác suất bỏ qua (dropout probability), tức là xác suất một unit (nơ-ron) sẽ bị loại bỏ ngẫu nhiên trong quá trình 
                huấn luyện. Đây là một giá trị số thực nằm trong khoảng từ 0 đến 1.
                Tham số inplace=True cho biết rằng phép dropout sẽ được thực hiện trực tiếp trên đầu vào mà không cần tạo ra một bản sao của nó. 
                Điều này có thể giúp tiết kiệm bộ nhớ và tăng tốc độ thực thi, bởi vì nó sẽ sử dụng trực tiếp bộ nhớ hiện có mà không cần phải tạo thêm 
                bộ nhớ phụ.
                Phép dropout giúp mô hình tránh việc phụ thuộc quá mức vào một số unit cụ thể trong quá trình huấn luyện, làm cho mô hình trở nên mạnh mẽ 
                hơn và tổng quát hóa tốt hơn đối với dữ liệu mới.
            nn.ReLU(inplace=True) if actx else nn.Identity()):
                Sử dụng để áp dụng hàm kích hoạt ReLU (Rectified Linear Unit) trong một mạng nơ-ron
                Tham số inplace=True: Nếu được đặt thành True, ReLU sẽ thay đổi đầu vào trực tiếp mà không cần tạo bản sao của nó. Điều này tiết kiệm 
                bộ nhớ và làm giảm thời gian thực thi. Tuy nhiên, nó cũng có thể dẫn đến việc mất mát dữ liệu không mong muốn nếu không sử dụng cẩn thận.
                nn.Identity(): Đây là một lớp đơn giản không làm thay đổi giá trị đầu vào và trả về đầu ra như là đầu vào của nó. Nó được sử dụng khi 
                không cần áp dụng bất kỳ biến đổi nào cho dữ liệu.
        """
        relu_lin = lambda a, b, dp, lnx, actx: nn.Sequential(
            nn.Linear(a, b),
            nn.LayerNorm(b, elementwise_affine=use_affine) if lnx else nn.Identity(),
            nn.Dropout(p=dp, inplace=True),
            nn.ReLU(inplace=True) if actx else nn.Identity())

        """
            Là một hàm lambda được định nghĩa để tạo ra một chuỗi các lớp Conv neural network (nn.Sequential).
            GCNConv(insize, outsize):
                Lớp này thực hiện phép tích chập trên dữ liệu đồ thị. Đầu vào của nó là insize, là kích thước của đặc trưng đầu vào của mỗi đỉnh trong 
                đồ thị. outsize là đặc trưng đầu ra của mỗi đỉnh sau khi áp dụng phép tích chập. GCNConv được thiết kế để học các biểu diễn đặc trưng 
                của đồ thị dựa trên cấu trúc liên kết giữa các đỉnh.
            GraphNorm(outsize):
                Sử dụng để thực hiện chuẩn hóa các đặc trưng (features) của đồ thị theo một cách tương tự như Layer Normalization nhưng được áp dụng 
                trên toàn bộ batch của các đỉnh trong đồ thị.
                Quá trình chuẩn hóa được áp dụng trên các đặc trưng của từng đỉnh trong đồ thị. Lớp này giúp cân bằng độ lớn của các đặc trưng của các 
                đỉnh trong một batch dữ liệu đồ thị. Quá trình chuẩn hóa giúp cho việc huấn luyện mô hình ổn định hơn bằng cách làm giảm hiện tượng 
                biến động quá mức (covariate shift) trong quá trình lan truyền ngược.
        """
        relu_conv = lambda insize, outsize, dp, act: Seq([
            GCNConv(insize, outsize),
            GraphNorm(outsize),
            Dropout(p=dp, inplace=True),
            nn.ReLU(inplace=True) if act else nn.Identity()
        ])

        self.max_x = max_x
        self.use_node_feat = use_node_feat
        self.node_feat = node_feat

        """
            Quá trình đồ thị học các đặc trưng của đỉnh:
                Nếu use_node_feat là True: Mô hình sẽ sử dụng các đặc trưng của đỉnh (node features).
                Nếu use_node_feat là False: Mô hình sẽ sử dụng một lớp nhúng (embedding layer) để biểu diễn các đỉnh.
        """
        if use_node_feat:
            self.lin1 = nn.Sequential(
                nn.Dropout(dp_lin0),
                relu_lin(node_feat.shape[-1], channels_1wl, dp_lin1, True, False)
            )
        else:
            """
                self.emb: Là một lớp tuần tự (Sequential) bao gồm:
                nn.Embedding(max_x + 1, channels_1wl): Là lớp nhúng (embedding layer) với số lượng đỉnh tối đa max_x + 1 và 
                    số chiều đặc trưng channels_1wl.
                GraphNorm(channels_1wl): Là lớp chuẩn hóa đồ thị để chuẩn hóa các đặc trưng của đỉnh.
                Dropout(p=dp_emb, inplace=True): Là lớp dropout áp dụng cho đầu ra của lớp nhúng.
            """
            self.emb = nn.Sequential(nn.Embedding(max_x + 1, channels_1wl),
                                     GraphNorm(channels_1wl),
                                     Dropout(p=dp_emb, inplace=True))
        """
            Khởi tạo một danh sách các moudle chức các lớp relu_conv
            channels_1wl và channels_2wl là các số chiều đầu vào và đầu ra của các lớp relu_conv
        """
        self.conv1s = nn.ModuleList(
            [relu_conv(channels_1wl, channels_1wl, dp_1wl0, act0) for _ in range(depth1 - 1)] +
            [relu_conv(channels_1wl, channels_2wl, dp_1wl1, act1)])

        self.conv2s = nn.ModuleList(
            [relu_conv(channels_2wl, channels_2wl, dp_2wl, True) for _ in range(depth2)])

        self.conv2s_r = nn.ModuleList(
            [relu_conv(channels_2wl, channels_2wl, dp_2wl, True) for _ in range(depth2)])
        """
            Xây dựng một lớp linear có vai trò biến đổi đầu vào với số chiều channels_2wl thành đầu ra với số chiều là 1. 
            Sau đó kq này được đưa vào hàm sigmoid() trong file train.train để đưa ra kq cuối cùng cho bài toán 
        """
        self.pred = nn.Linear(channels_2wl, 1)

    def forward(self, x, edge1, pos, idx=None, ei2=None, test=False):
        """
            :x = x_new
            :edge1 = ei_new
            :pos = dataset.pos1
            :idx = pos2
            :ei2 = ei2_new
        """
        edge2, edge2_r = reverse(ei2)

        x = self.lin1(self.node_feat) if self.use_node_feat else self.emb(x).squeeze()
        for conv1 in self.conv1s:
            x = conv1(x, edge1)

        x = x[pos[:, 0]] * x[pos[:, 1]]
        for i in range(len(self.conv2s)):
            x = self.conv2s[i](x, edge2) + self.conv2s_r[i](x, edge2_r)
        x = x[idx]
        mask = torch.cat(
            [torch.ones([1, x.shape[0] // 2], dtype=bool),
             torch.zeros([1, x.shape[0] // 2], dtype=bool)]).t().reshape(-1)
        x = x[mask] * x[~mask]
        x = self.pred(x)
        return x


class Seq(nn.Module):
    def __init__(self, modlist):
        super().__init__()
        self.modlist = nn.ModuleList(modlist)

    def forward(self, *args, **kwargs):
        out = self.modlist[0](*args, **kwargs)
        for i in range(1, len(self.modlist)):
            out = self.modlist[i](out)
        return out


def mataggr(A, h, g):
    '''
    A (n, n, d). n is number of node, d is latent dimension
    h, g are mlp
    '''
    B = h(A)
    # C = f(A)
    n, d = A.shape[0], A.shape[1]
    vec_p = (torch.sum(B, dim=1, keepdim=True)).expand(-1, n, -1)
    vec_q = (torch.sum(B, dim=0, keepdim=True)).expand(n, -1, -1)
    D = torch.cat([A, vec_p, vec_q], -1)
    return g(D)
