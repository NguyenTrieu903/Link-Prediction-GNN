import time

import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from TwoWL.utils import sample_block, double
from assets.theme import *


def train(mod, opt, dataset, batch_size, i):
    mod.train()

    global perm1, perm2, pos_batchsize, neg_batchsize
    if i == 0:
        pos_batchsize = batch_size // 2
        neg_batchsize = batch_size // 2
        perm1 = torch.randperm(dataset.ei.shape[1] // 2, device=dataset.x.device)
        perm2 = torch.randperm((dataset.pos1.shape[0] - dataset.ei.shape[1]) // 2,
                               device=dataset.x.device)

    idx1 = perm1[i * pos_batchsize:(i + 1) * pos_batchsize]
    idx2 = perm2[i * neg_batchsize:(i + 1) * neg_batchsize]

    y = torch.cat((torch.ones_like(idx1, dtype=torch.float),
                   torch.zeros_like(idx2, dtype=torch.float)),
                  dim=0).unsqueeze(-1)

    idx1 = double(idx1, for_index=True)
    idx2 = double(idx2, for_index=True) + dataset.ei.shape[1]

    ei_new, x_new, ei2_new = sample_block(idx1, dataset.x.shape[0], dataset.ei, dataset.ei2)
    pos2 = torch.cat((idx1, idx2), dim=0)

    opt.zero_grad()
    pred = mod(x_new, ei_new, dataset.pos1, pos2, ei2_new)
    loss = F.binary_cross_entropy_with_logits(pred, y)
    loss.backward()
    opt.step()

    with torch.no_grad():
        sig = pred.sigmoid().cpu().numpy()
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
    sig = pred.sigmoid().cpu()
    mask = torch.cat(
        [torch.ones([1, sig.shape[0]], dtype=bool), torch.zeros([1, sig.shape[0]], dtype=bool)]).t().reshape(
        -1, 1)

    result = roc_auc_score(dataset.y[mask].squeeze().cpu().numpy(), sig)
    fpr, tpr, thresholds = roc_curve(dataset.y[mask].squeeze().cpu().numpy(), sig)
    return result, fpr, tpr


def train_routine(dsname, mod, opt, trn_ds, val_ds, tst_ds, epoch, verbose=True):
    def vprint(*args, **kwargs):
        if verbose:
            print(*args, **kwargs)

    trn_ds.pos1 = trn_ds.pos1.to(torch.long)
    val_ds.pos1 = val_ds.pos1.to(torch.long)
    tst_ds.pos1 = tst_ds.pos1.to(torch.long)
    batch_size = val_ds.y.shape[0]
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
    if verbose:
        with open(PATH_SAVE_TEST_AUC + f'{dsname}_auc_record_twowl.txt', 'a') as f:
            f.write('AUC:' + str(round(tst_score, 4)) + '   ' + 'Time:' + str(
                round(t1 - t0, 4)) + '   ' + '\n')

        values_auc = []
        annotations_auc = []
        with open(PATH_SAVE_TEST_AUC + f'{dsname}_auc_record_twowl.txt', 'r') as f1:
            auc = f1.readlines()
        if auc:
            for line in auc:
                line = line.strip()
                if line:
                    AUC, times = line.split()
                    # x_txt.append(len(x_txt) + 1)
                    values_auc.append(float(AUC.split(":")[1]))
                    annotations_auc.append(float(times.split(":")[1]))
            if tst_score >= max(values_auc):
                fpr_file = "fpr.json"
                tpr_file = "tpr.json"

                with open(fpr_file, "w") as f:
                    json.dump(fpr.tolist(), f)

                with open(tpr_file, "w") as f:
                    json.dump(tpr.tolist(), f)
    return best_val
