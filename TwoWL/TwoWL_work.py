import random
import torch
from torch.optim import Adam
import yaml
import optuna
import json
from TwoWL.operators.datasets import load_dataset, dataset
from TwoWL.model import train
from TwoWL.model.model import LocalWLNet, WLNet, FWLNet, LocalFWLNet


def work(args, device="cpu"):
    device = torch.device(device)
    bg = load_dataset("2wl_l")
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
            bg = load_dataset("2wl_l")
            bg.to(device)
            bg.preprocess()
            bg.setPosDegreeFeature()
            trn_ds = dataset(*bg.split(0))
            val_ds = dataset(*bg.split(1))
            tst_ds = dataset(*bg.split(2))
        lr = trial.suggest_categorical("lr", [0.0005, 0.001, 0.005, 0.01, 0.05])
        depth1 = trial.suggest_int("l1", 1, 3)
        depth2 = trial.suggest_int("l2", 1, 3)
        channels_1wl = trial.suggest_categorical("h1", [24, 32, 64])
        channels_2wl = trial.suggest_categorical("h2", [16, 24])
        dp_lin0 = trial.suggest_float("dpl0", 0.0, 0.8, step=0.1)
        dp_lin1 = trial.suggest_float("dpl1", 0.0, 0.8, step=0.1)
        dp_emb = trial.suggest_float("dpe", 0.0, 0.5, step=0.1)
        dp_1wl0 = trial.suggest_float("dp10", 0.0, 0.5, step=0.1)
        dp_1wl1 = trial.suggest_float("dp11", 0.0, 0.5, step=0.1)
        dp_2wl = trial.suggest_float("dp2", 0.0, 0.5, step=0.1)
        act0 = trial.suggest_categorical("a1", [True, False])
        act1 = trial.suggest_categorical("a2", [True, False])
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
        mod = LocalWLNet(max_degree, use_node_attr, trn_ds.na).to(device)
        for param_tensor in mod.state_dict():
            print(param_tensor, "\t", mod.state_dict()[param_tensor].size())
        torch.save(mod.state_dict(), 'model.pkl')
        opt = Adam(mod.parameters(), lr=lr)
        return train.train_routine("fb-pages-food", mod, opt, trn_ds, val_ds, tst_ds, epoch, verbose=True)


    # valparam()
    study = optuna.create_study(direction='maximize')
    study.optimize(selparam, n_trials=10)  # Tối ưu hoá với 100 thử nghiệm
    best_params = study.best_params

    with open("logs.yaml", "w") as f:
        json.dump(best_params, f)

def test(kwargs):
    device = torch.device("cpu")
    bg = load_dataset("2wl_l")
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
    # model = torch.load('model.pkl')
    mod = LocalWLNet(max_degree, use_node_attr, tst_ds.na, **kwargs).to(device)
    return train.test(mod, tst_ds)
def call_back_test():
    with open(f"logs.yaml") as f:
        params = yaml.safe_load(f)
    print(params)
    test(**(params))
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
    # work(args)
    call_back_test()