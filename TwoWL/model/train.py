import time

import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from TwoWL.utils import sample_block, double
from assets.theme import *


def train(mod, opt, dataset, batch_size, i):
    """
        Phương thức mod.train() được sử dụng để đặt mô hình mod vào chế độ huấn luyện (training mode) trong PyTorch.
        Đây là một bước quan trọng trong quá trình huấn luyện mô hình.
        Training Mode:
            Khi mô hình được đặt vào chế độ huấn luyện bằng cách gọi mod.train(), các lớp như dropout hoặc batch normalization (nếu có) sẽ hoạt
            động theo cách đã thiết lập để học và cập nhật tham số.
            Dropout: Các đơn vị trong mạng nơ-ron sẽ được ngẫu nhiên bỏ qua để ngăn chặn sự phụ thuộc quá mức vào một số đơn vị cụ thể.
            Batch Normalization: Chuẩn hóa lại giá trị đầu ra của mỗi lớp, giúp cho việc huấn luyện hiệu quả hơn.
    """
    mod.train()
    """
        Quá trình thiết lập dữ liệu để train 
    """
    global perm1, perm2, pos_batchsize, neg_batchsize
    if i == 0:
        pos_batchsize = batch_size // 2
        neg_batchsize = batch_size // 2
        print("dataset.ei.shape[1] ", dataset.ei.shape[1])
        print("dataset.pos1.shape[0] ", dataset.pos1.shape[0])
        print("dataset.pos1.shape ", dataset.pos1.size())
        print("dataset.ei.shape ", dataset.ei.size())

        perm1 = torch.randperm(dataset.ei.shape[1] // 2, device=dataset.x.device)
        perm2 = torch.randperm((dataset.pos1.shape[0] - dataset.ei.shape[1]) // 2,
                               device=dataset.x.device)

    """
        Dùng để trích xuất các chỉ số của cạnh (edge indices) từ các permutated indices (chỉ số được hoán vị) perm1 và perm2, 
        để tạo ra các batch nhỏ hơn (mini-batches) cho các cạnh dương (positive edges) và các cạnh âm (negative edges) trong quá trình huấn 
        luyện hoặc dự đoán.
        Vd:
            perm1 là tensor [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            perm2 là tensor [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
            pos_batchsize = 2
            neg_batchsize = 3
            i = 1
            
            idx1 = perm1[1 * 2:(1 + 1) * 2]
            => idx1 = perm1[2:4]
            => idx1 = tensor([2, 3])
            idx2 = perm2[1 * 3:(1 + 1) * 3]
            => idx2 = perm2[3:6]
            => idx2 = tensor([13, 14, 15])
        Batching: Việc chia nhỏ dữ liệu thành các batch giúp cho quá trình huấn luyện mô hình trở nên hiệu quả hơn, đặc biệt khi làm việc với các 
            tập dữ liệu lớn. Điều này cũng giúp giảm bớt yêu cầu về bộ nhớ và có thể dẫn đến quá trình huấn luyện ổn định hơn.
        Dữ liệu dương và âm: Trích xuất các batch của các cạnh dương và cạnh âm một cách riêng biệt giúp mô hình học được cách phân biệt giữa 
            các cạnh tồn tại và không tồn tại trong đồ thị.
    """
    idx1 = perm1[i * pos_batchsize:(i + 1) * pos_batchsize]
    idx2 = perm2[i * neg_batchsize:(i + 1) * neg_batchsize]

    """
        Dùng để tạo ra một tensor y chứa các nhãn (labels) cho các cạnh (edges) trong bài toán dự đoán liên kết (link prediction). 
        Cụ thể, tensor này kết hợp các nhãn dương (positive labels) và nhãn âm (negative labels) thành một tensor duy nhất và thêm một chiều mới 
        ở cuối tensor.
        Kết hợp hai tensor trên dọc theo chiều 0 (chiều hàng) để tạo ra một tensor mới chứa cả nhãn dương và nhãn âm.
        Kết quả là một tensor 1D với các giá trị 1 ở đầu (cho các cạnh dương) và các giá trị 0 ở sau (cho các cạnh âm).
        unsqueeze(-1)
            Thêm một chiều mới ở cuối tensor (chiều -1) để tensor trở thành một tensor cột 2D. 
            Điều này có thể hữu ích khi bạn cần tensor có định dạng (num_samples, 1).
    """
    y = torch.cat((torch.ones_like(idx1, dtype=torch.float),
                   torch.zeros_like(idx2, dtype=torch.float)),
                  dim=0).unsqueeze(-1)

    """
        Mở rộng dữ liệu về cạnh
    """
    idx1 = double(idx1, for_index=True)
    idx2 = double(idx2, for_index=True) + dataset.ei.shape[1]

    ei_new, x_new, ei2_new = sample_block(idx1, dataset.x.shape[0], dataset.ei, dataset.ei2)
    pos2 = torch.cat((idx1, idx2), dim=0)
    """
        Quá trình tính loss (Binary Cross Entropy) và cập nhật tham số (Gradient descent)
    """
    """
        Phương thức này được sử dụng để đặt gradient của tất cả các tham số trong mô hình về 0. 
        Trong PyTorch, gradient của mỗi tham số được tích lũy sau mỗi lần lan truyền ngược (backward propagation). 
        Trước khi tính toán gradient mới cho một lần lan truyền ngược mới, 
        bạn cần đảm bảo rằng gradient của các tham số đã được xóa để tránh sự tích lũy không mong muốn. 
        Điều này thường được thực hiện ngay trước khi gọi loss.backward() để tính toán gradient của hàm mất mát.
    """
    opt.zero_grad()
    pred = mod(x_new, ei_new, dataset.pos1, pos2, ei2_new)
    """ 
    loss = F.binary_cross_entropy_with_logits(pred, y): 
    Dùng để tính toán giá trị hàm mất mát sử dụng BCE with logits giữa các dự đoán của mô hình và nhãn thực tế. 
    Giá trị loss này sau đó được sử dụng để tính gradient trong quá trình lan truyền ngược (backpropagation) và cập nhật các tham số của mô hình 
    trong quá trình huấn luyện.
    (BCE with logits) là phiên bản tối ưu hóa của hàm Binary Cross Entropy
        Binary Cross Entropy: 
            Là một trong những hàm mất mát phổ biến trong bài toán phân loại nhị phân.
            Nó đo lường sự khác biệt giữa các giá trị dự đoán (sử dụng hàm sigmoid) và nhãn thực tế.
        pred: Là đầu ra của mô hình mạng nơ-ron trước khi áp dụng hàm sigmoid, được tính dựa trên dữ liệu đầu vào.
        y: Là nhãn thực tế của dữ liệu đầu vào.
    """
    loss = F.binary_cross_entropy_with_logits(pred, y)
    """
    Gradient Descent và Backpropagation:
    Trong quá trình huấn luyện mô hình học sâu, mục tiêu là điều chỉnh các tham số của mô hình để giảm thiểu sai số giữa giá trị dự đoán và giá trị 
    thực tế. Để làm được điều này, chúng ta cần tính gradient của hàm mất mát (loss function) theo từng tham số của mô hình.
    Phương pháp Backpropagation (lan truyền ngược) là một cách hiệu quả để tính gradient này. Nó hoạt động từ phía cuối mạng nơ-ron (output layer) 
    trở về phía đầu mạng (input layer), tính toán gradient của hàm mất mát theo từng lớp và từng tham số.
    
    Khi gọi loss.backward(), PyTorch sẽ tính toán gradient của loss theo các tham số của mô hình.
    là phương thức để tính toán gradient của hàm mất mát (loss) theo các tham số của mô hình.
    """
    loss.backward()
    """
    Sau khi tính toán gradient từ loss.backward() và đã đảm bảo rằng gradient đã được cập nhật, bạn gọi opt.step() 
    để thực hiện bước cập nhật tham số của mô hình. Phương thức này sẽ cập nhật các tham số bằng cách sử dụng gradient đã tính được trong backward 
    và tỷ lệ học tập (lr) đã được cấu hình trong Adam (hoặc bộ tối ưu hóa khác).
    """
    opt.step()
    """
    with torch.no_grad(): báo cho PyTorch biết rằng các phép tính bên trong khối này không cần phải theo dõi và tính toán gradient. 
    Trong quá trình đánh giá mô hình, chúng ta thường không cần cập nhật các tham số mô hình, chỉ cần tính toán kết quả dự đoán và đánh giá hiệu suất 
    mô hình. Việc sử dụng torch.no_grad() giúp tăng tốc độ tính toán bằng cách bỏ qua quá trình tính gradient và lưu trữ.
    """
    with torch.no_grad():
        """
        Đoạn code này chuyển đổi kết quả dự đoán (pred) từ đầu ra của mạng nơ-ron thành xác suất bằng hàm sigmoid (sigmoid()), 
        sau đó chuyển về CPU và chuyển thành mảng NumPy (numpy())
        """
        sig = pred.sigmoid().cpu().numpy()
        """
        Dùng để tính toán điểm ROC AUC (Area Under the ROC Curve) để đánh giá hiệu suất của mô hình dựa trên dự đoán (sig) so với nhãn thực tế (y). 
        ROC AUC là một phép đo đánh giá khả năng phân loại của mô hình, đo lường tỷ lệ của dương tính thực sự được phân loại đúng so với tỷ lệ của giả 
        dương tính bị phân loại nhầm.
        """
        score = roc_auc_score(y.cpu().numpy(), sig)
    i += 1
    if (i + 1) * pos_batchsize > perm1.shape[0]:
        i = 0
    return loss.item(), score, i


@torch.no_grad()
def test(mod, dataset, test=False):
    mod.eval()
    pred = mod(
        dataset.x,
        dataset.ei,
        dataset.pos1,
        dataset.ei.shape[1] + torch.arange(dataset.y.shape[0], device=dataset.x.device),
        dataset.ei2,
        True)
    print("pred size:", pred.size())
    sig = pred.sigmoid().cpu()
    mask = torch.cat(
        [torch.ones([1, sig.shape[0]], dtype=bool), torch.zeros([1, sig.shape[0]], dtype=bool)]).t().reshape(
        -1, 1)

    result = roc_auc_score(dataset.y[mask].squeeze().cpu().numpy(), sig)
    fpr, tpr, thresholds = roc_curve(dataset.y[mask].squeeze().cpu().numpy(), sig)
    print("result", result)
    return result, fpr, tpr


def train_routine(dsname, mod, opt, trn_ds, val_ds, tst_ds, epoch, verbose=True):
    print("calling train_routine")

    def vprint(*args, **kwargs):
        if verbose:
            print(*args, **kwargs)

    trn_ds.pos1 = trn_ds.pos1.to(torch.long)
    val_ds.pos1 = val_ds.pos1.to(torch.long)
    tst_ds.pos1 = tst_ds.pos1.to(torch.long)
    batch_size = trn_ds.y.shape[0]
    vprint(f"batch size{batch_size}")

    best_val = 0
    tst_score = 0
    early_stop = 0
    early_stop_thd = 800
    for i in range(epoch):
        train_idx = 0
        t0 = time.time()
        loss, trn_score, train_idx = train(mod, opt, trn_ds, batch_size, train_idx)
        t1 = time.time()
        val_score, fpr, tpr = test(mod, val_ds)
        vprint(f"epoch: {i:03d}, trn: time {t1 - t0:.2f} s, loss {loss:.4f}, trn {trn_score:.4f}, val {val_score:.4f}",
               end=" ")
        early_stop += 1
        if val_score > best_val:
            early_stop = 0
            best_val = val_score
            if verbose:
                t0 = time.time()
                tst_score, fpr, tpr = test(mod, tst_ds, True)
                t1 = time.time()
                # vprint(f"time:{t1-t0:.4f}")
            vprint(f"tst {tst_score:.4f}")
        else:
            vprint()
        if early_stop > early_stop_thd:
            break
    vprint(f"end test {tst_score:.3f}")
    # if verbose:
    #     # with open(f'TwoWL/records/{dsname}_auc_record.txt', 'a') as f:
    #     with open(PATH_SAVE_TEST_AUC + f'{dsname}_auc_record_twowl.txt', 'a') as f:
    #         f.write('AUC:' + str(round(tst_score, 4)) + '   ' + 'Time:' + str(
    #             round(t1 - t0, 4)) + '   ' + '\n')
    #
    #     values_auc = []
    #     annotations_auc = []
    #     with open(PATH_SAVE_TEST_AUC + f'{dsname}_auc_record_twowl.txt', 'r') as f1:
    #         auc = f1.readlines()
    #     if auc:
    #         for line in auc:
    #             line = line.strip()
    #             if line:
    #                 AUC, times = line.split()
    #                 # x_txt.append(len(x_txt) + 1)
    #                 values_auc.append(float(AUC.split(":")[1]))
    #                 annotations_auc.append(float(times.split(":")[1]))
    #         if tst_score >= max(values_auc):
    #             fpr_file = "fpr.json"
    #             tpr_file = "tpr.json"
    #
    #             with open(fpr_file, "w") as f:
    #                 json.dump(fpr.tolist(), f)
    #
    #             with open(tpr_file, "w") as f:
    #                 json.dump(tpr.tolist(), f)
    return best_val
