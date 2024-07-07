import json
import random
import time

import numpy as np
import optuna
import streamlit as st
import torch
from torch.optim import Adam

from TwoWL.model import train
from TwoWL.model.model import LocalWLNet
from TwoWL.operators.datasets import load_dataset, dataset
from assets.theme import update_time
from constant import *


def work(args, device="cpu"):
    global seconds_passed
    seconds_passed = 0

    total_steps = 4
    step_size = 100 / total_steps
    current_step = 0

    progress_bar = st.sidebar.progress(current_step)
    status_text = st.sidebar.empty()
    time_display = st.sidebar.empty()
    status_text.text("0% Complete")

    start_time = time.time()
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

    current_step += step_size
    status_text.text("{:.0f}% Complete".format(current_step))
    progress_bar.progress(current_step / 100)
    update_time(time_display, start_time, time.time())

    def selparam(trial):
        nonlocal bg, trn_ds, val_ds, tst_ds
        time_start = time.time()
        if random.random() < 0.1:
            bg = load_dataset(args.pattern)
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

        return valparam(setting, time_start, trial.number)

    def valparam(kwargs, time_start, trial_number):
        lr = kwargs.pop('lr')
        epoch = args.epoch
        mod = LocalWLNet(max_degree, use_node_attr, trn_ds.na, **kwargs).to(device)
        opt = Adam(mod.parameters(), lr=lr)
        model = train.train_routine("fb-pages-food", mod, opt, trn_ds, val_ds, tst_ds, epoch, verbose=True)
        time_end = time.time()

        with open(PATH_TIME_TWOWL + 'time_twowl.txt', 'a') as f:
            f.write('Time:' + str(round(time_end - time_start, 4)) + '\n')

        return model

    start_time = time.time()
    study = optuna.create_study(direction='maximize')
    current_step += step_size
    status_text.text("{:.0f}% Complete".format(current_step))
    progress_bar.progress(current_step / 100)
    update_time(time_display, start_time, time.time())

    start_time = time.time()
    progress_ = st.progress(0)
    status_ = st.empty()
    status_.text("0% Complete the training process")
    chart = st.line_chart()

    def callback(study, trial):
        status_.text("{:.0f}% Complete the training process".format((trial.number + 1) * 10))  # Hiển thị tiến độ
        new_rows = np.full((1, 1), trial.value)
        chart.add_rows(new_rows)
        progress_.progress((trial.number + 1) * 10)  # Hiển thị tiến độ trên thanh tiến trình
        time.sleep(0.01)  # Đợi 0.01 giây để mô phỏng quá trình huấn luyện

    study.optimize(selparam, n_trials=10,
                   callbacks=[lambda study, trial: callback(study, trial)])  # Tối ưu hoá với 100 thử nghiệm
    current_step += step_size
    status_text.text("{:.0f}% Complete".format(current_step))
    progress_bar.progress(current_step / 100)
    update_time(time_display, start_time, time.time())

    start_time = time.time()
    # Tên tệp nhật ký để lưu trữ thông số
    log_file = "logs.json"
    best_params = study.best_params

    with open(log_file, "w") as f:
        json.dump(best_params, f)

    current_step += step_size
    status_text.text("{:.0f}% Complete".format(current_step))
    progress_bar.progress(current_step / 100)
    update_time(time_display, start_time, time.time())


def read_results_twowl():
    auc_twowl = "fb-pages-food_auc_record_twowl.txt"
    with open("logs.json", "r") as f1, open(PATH_SAVE_TEST_AUC + auc_twowl, "r") as f2, open(
            PATH_TIME_TWOWL + "time_twowl.txt", "r") as f3:
        logs = json.load(f1)
        auc = f2.readlines()
        time_twowl = f3.readlines()

    best_auc_twowl = 0.0
    for line in auc:
        line = line.strip()
        if line:
            AUC, time = line.split()
            AUC = float(AUC.split(":")[1])
            if AUC >= best_auc_twowl:
                best_auc_twowl = AUC

    time_train = []
    for line in time_twowl:
        line = line.strip()
        if line:
            time_value = float(line.split(":")[1])
            time_train.append(time_value)
    average_time = sum(time_train) / len(time_train)
    return logs, best_auc_twowl, average_time
