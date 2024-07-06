import json
import random

import optuna
import torch
import yaml
from torch.optim import Adam

from TwoWL.model import train
from TwoWL.model.model import LocalWLNet
from TwoWL.operators.datasets import load_dataset, dataset


def work(args, device="cpu"):
    device = torch.device(device)
    bg = load_dataset("2wl_l")
    bg.to(device)
    # Hàm này dùng để tách ra các tập train, validation và test sau khi đã load và xử lý dữ liệu
    bg.preprocess()
    # Hàm này dùng để tính ma trận bậc của các nút
    bg.setPosDegreeFeature()
    max_degree = torch.max(bg.x[0])
    """
        Mỗi tập dữ liệu bao gồm:
        :x - Ma trận số bậc của các nút trong tập 
        :na - None 
        :ei - (edge_index) index của các nút trong tập 
        :ea - edge_attribute thuộc tính của các nút trong tập (ở đây là none)
        :pos1s - positive edges set một tập hợp các edge positive và các pred_edge dùng để dự đoán 
        :y - label cho các edge. 1 là positive và 0 là negative 
        :ei2 - edge index matrix ma trận chỉ số edge. Mỗi edge trong data đều có 1 chỉ số  
    """
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
            bg = load_dataset("2wl_l")
            bg.to(device)
            bg.preprocess()
            bg.setPosDegreeFeature()
            trn_ds = dataset(*bg.split(0))
            val_ds = dataset(*bg.split(1))
            tst_ds = dataset(*bg.split(2))
        lr = trial.suggest_categorical("lr", [0.0005, 0.001, 0.005, 0.01, 0.05])
        depth1 = trial.suggest_int("depth1", 1, 3)
        depth2 = trial.suggest_int("depth2", 1, 3)
        channels_1wl = trial.suggest_categorical("channels_1wl", [24, 32, 64])
        channels_2wl = trial.suggest_categorical("channels_2wl", [16, 24])
        dp_lin0 = trial.suggest_float("dp_lin0", 0.0, 0.8, step=0.1)
        dp_lin1 = trial.suggest_float("dp_lin1", 0.0, 0.8, step=0.1)
        dp_emb = trial.suggest_float("dp_emb", 0.0, 0.5, step=0.1)
        dp_1wl0 = trial.suggest_float("dp_1wl0", 0.0, 0.5, step=0.1)
        dp_1wl1 = trial.suggest_float("dp_1wl1", 0.0, 0.5, step=0.1)
        dp_2wl = trial.suggest_float("dp_2wl", 0.0, 0.5, step=0.1)
        act0 = trial.suggest_categorical("act0", [True, False])
        act1 = trial.suggest_categorical("act1", [True, False])
        setting = {
            'dp_lin0': dp_lin0,
            'dp_lin1': dp_lin1,
            'dp_emb': dp_emb,
            'dp_1wl0': dp_1wl0,
            'dp_1wl1': dp_1wl1,
            'dp_2wl': dp_2wl,
            'channels_1wl': channels_1wl,
            'channels_2wl': channels_2wl,
            'depth1': depth1,
            'depth2': depth2,
            'act0': act0,
            'act1': act1,
            'lr': lr,
        }

        return valparam(setting)

    # train to find best_param (save to file) and save model. After load model with best_params to test data
    def valparam(kwargs):
        lr = kwargs.pop('lr')
        epoch = 1000
        mod = LocalWLNet(max_degree, use_node_attr, trn_ds.na, **kwargs).to(device)
        # for param_tensor in mod.state_dict():
        #     print(param_tensor, "\t", mod.state_dict()[param_tensor].size())

        # save model
        torch.save(mod.state_dict(), 'model.pkl')
        # tối ưu hóa tham số dùng hàm Adam
        opt = Adam(mod.parameters(), lr=lr)
        return train.train_routine("fb-pages-food", mod, opt, trn_ds, val_ds, tst_ds, epoch, verbose=True)

    # valparam()
    study = optuna.create_study(direction='maximize')
    study.optimize(selparam, n_trials=10)  # Tối ưu hoá với 100 thử nghiệm
    best_params = study.best_params

    with open("logs.yaml", "w") as f:
        json.dump(best_params, f)


def test(kwargs):
    print("kwargs ", kwargs)
    device = torch.device("cpu")
    bg = load_dataset("2wl_l")
    bg.to(device)
    bg.preprocess()
    bg.setPosDegreeFeature()
    max_degree = torch.max(bg.x[0])

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
    mod = LocalWLNet(max_degree, use_node_attr, tst_ds.na, **kwargs).to(device)
    # load model
    state_dict = torch.load('model.pkl')
    model_state_dict = mod.state_dict()
    filtered_state_dict = {k: v for k, v in state_dict.items() if
                           k in model_state_dict and v.size() == model_state_dict[k].size()}
    model_state_dict.update(filtered_state_dict)
    mod.load_state_dict(model_state_dict)
    return train.test(mod, tst_ds)


def call_back_test():
    with open(f"logs.yaml") as f:
        params = yaml.safe_load(f)
    if 'lr' in params:
        del params['lr']
    print("params ", params)
    test(params)


def load_model():
    model = torch.load('model.pkl')
    for key, value in model.items():
        print(f'{key}: {value}')
    return model


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
    work(args)
    # load_model()
    # call_back_test()
