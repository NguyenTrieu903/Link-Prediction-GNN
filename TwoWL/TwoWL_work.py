import random
import torch
from torch.optim import Adam

import optuna
import json
from TwoWL.operators.datasets import load_dataset, dataset
from TwoWL.model import train
from TwoWL.model.model import LocalWLNet, WLNet, FWLNet, LocalFWLNet


def work(args, device="cpu"):
    device = torch.device(device)
    bg = load_dataset(args.pattern)
    bg.to(device)
    bg.preprocess()
    bg.setPosDegreeFeature()
    max_degree = torch.max(bg.x[2])

    tst_ds = dataset(*bg.split(2))
    trn_ds = dataset(*bg.split(0))
    val_ds = dataset(*bg.split(1))
    if trn_ds.na != None:
        print("use node feature")
        trn_ds.na = trn_ds.na.to(device)
        val_ds.na = val_ds.na.to(device)
        tst_ds.na = tst_ds.na.to(device)
        use_node_attr = True
    else:
        use_node_attr = False

    def selparam(trial):
        nonlocal bg, trn_ds, val_ds, tst_ds
        if random.random() < 0.1:
            bg = load_dataset(args.pattern)
            bg.to(device)
            bg.preprocess()
            bg.setPosDegreeFeature()
            trn_ds = dataset(*bg.split(0))
            val_ds = dataset(*bg.split(1))
            tst_ds = dataset(*bg.split(2))
        lr = trial.suggest_categorical("lr", [0.0005, 0.001, 0.005, 0.01, 0.05])
        layer1 = trial.suggest_int("layer1", 2, 3)
        layer2 = trial.suggest_int("layer2", 1, 3)
        layer3 = trial.suggest_int("layer3", 1, 3)
        hidden_dim_1 = trial.suggest_categorical("h1", [20])
        hidden_dim_2 = trial.suggest_categorical("h1", [20])
        dp0_0 = trial.suggest_float("dp0_0", 0.0, 0.8, step=0.1)
        dp0_1 = trial.suggest_float("dp0_1", 0.0, 0.8, step=0.1)
        dp1 = trial.suggest_float("dp1", 0.0, 0.5, step=0.1)
        dp2 = trial.suggest_float("dp2", 0.0, 0.5, step=0.1)
        dp3 = trial.suggest_float("dp3", 0.0, 0.5, step=0.1)
        ln0 = trial.suggest_categorical("ln0", [True, False])
        ln1 = trial.suggest_categorical("ln1", [True, False])
        ln2 = trial.suggest_categorical("ln2", [True, False])
        ln3 = trial.suggest_categorical("ln3", [True, False])
        ln4 = trial.suggest_categorical("ln4", [True, False])
        act0 = trial.suggest_categorical("a0", [True, False])
        act1 = trial.suggest_categorical("a1", [True, False])
        act2 = trial.suggest_categorical("a2", [True, False])
        act3 = trial.suggest_categorical("a3", [True, False])
        act4 = trial.suggest_categorical("a4", [True, False])
        setting = {
            'hidden_dim_1':hidden_dim_1,
            'hidden_dim_2':hidden_dim_2,
            'layer1': layer1,
            'layer2': layer2,
            'layer3': layer3,
            'dp0_0': dp0_0,
            'dp0_1': dp0_1,
            'dp1': dp1,
            'dp2': dp2,
            'dp3': dp3,
            'ln0': ln0,
            'ln1': ln1,
            'ln2': ln2,
            'ln3': ln3,
            'ln4': ln4,
            'act0': act0,
            'act1': act1,
            'act2': act2,
            'act3': act3,
            'act4': act4,
            'lr': lr,
        }
        return valparam(setting)

    def valparam(kwargs):
        lr = kwargs.pop('lr')
        epoch = args.epoch
        if args.pattern == '2wl':
            mod = WLNet(max_degree, use_node_attr, trn_ds.na, **kwargs).to(device)
        opt = Adam(mod.parameters(), lr=lr)
        return train.train_routine("fb-pages-food", mod, opt, trn_ds, val_ds, tst_ds, epoch, verbose=True)

    study = optuna.create_study(direction='maximize')
    study.optimize(selparam, n_trials=1)  # Tối ưu hoá với 100 thử nghiệm
    best_params = study.best_params

    # Tên tệp nhật ký để lưu trữ thông số
    log_file = "logs.json"

    # Ghi thông số vào tệp nhật ký
    with open(log_file, "w") as f:
        json.dump(best_params, f)

    print("Các thông số tối ưu đã được lưu vào tệp nhật ký:", log_file)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dataset', type=str, default="fb-pages-food")
    parser.add_argument('--pattern', type=str, default="2wl_l")
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--episode', type=int, default=200)

    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=int, default=-1)
    parser.add_argument('--path', type=str, default="Opt/")
    parser.add_argument('--test', action="store_true")
    parser.add_argument('--check', action="store_true")
    args = parser.parse_args()
    if args.device < 0:
        args.device = "cpu"
    else:
        args.device = "cuda:" + str(args.device)
    work(args.device)