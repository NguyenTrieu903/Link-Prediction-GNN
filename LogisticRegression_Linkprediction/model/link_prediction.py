import time

import streamlit as st

from LogisticRegression_Linkprediction.model import build_model
from LogisticRegression_Linkprediction.model.build_model import split_data
from LogisticRegression_Linkprediction.operators import dataset_preparation, understanding_data
from LogisticRegression_Linkprediction.operators.understanding_data import load_data
from LogisticRegression_Linkprediction.utils.feature_extraction import feature_extraction
from assets.theme import update_time
from constant import *


def link_prediction_with_logistic():
    global seconds_passed
    seconds_passed = 0

    total_steps = 8
    step_size = 100 / total_steps
    current_step = 0

    progress_bar = st.sidebar.progress(current_step)
    status_text = st.sidebar.empty()
    time_display = st.sidebar.empty()
    status_text.text("0% Complete")

    # Gọi hàm load_data()
    start_time = time.time()
    fb_df, node_list_1, node_list_2 = load_data()
    current_step += step_size
    status_text.text("{:.0f}% Complete".format(current_step))
    progress_bar.progress(current_step / 100)  # Chia cho 100 ở đây
    update_time(time_display, start_time, time.time())

    # Tạo đồ thị
    start_time = time.time()
    G = understanding_data.create_graph(fb_df)
    current_step += step_size
    status_text.text("{:.0f}% Complete".format(current_step))
    progress_bar.progress(current_step / 100)
    update_time(time_display, start_time, time.time())

    # Chuẩn bị dữ liệu
    start_time = time.time()
    data = dataset_preparation.retrieve_unconnected(node_list_1, node_list_2, G)
    current_step += step_size
    status_text.text("{:.0f}% Complete".format(current_step))
    progress_bar.progress(current_step / 100)
    update_time(time_display, start_time, time.time())

    start_time = time.time()
    omissible_links_index = dataset_preparation.remove_link_connected(fb_df, G)
    data, fb_df_ghost = dataset_preparation.data_for_model_training(fb_df, omissible_links_index, data)
    current_step += step_size
    status_text.text("{:.0f}% Complete".format(current_step))
    progress_bar.progress(current_step / 100)
    update_time(time_display, start_time, time.time())

    start_time = time.time()
    data, fb_df_ghost = dataset_preparation.data_for_model_training(fb_df, omissible_links_index, data)
    current_step += step_size
    status_text.text("{:.0f}% Complete".format(current_step))
    progress_bar.progress(current_step / 100)
    update_time(time_display, start_time, time.time())

    # Trích xuất đặc trưng
    start_time = time.time()
    x = feature_extraction(fb_df, fb_df_ghost, data)
    current_step += step_size
    status_text.text("{:.0f}% Complete".format(current_step))
    progress_bar.progress(current_step / 100)
    update_time(time_display, start_time, time.time())

    # Chia dữ liệu thành tập train và test
    start_time = time.time()
    xtrain, xtest, ytrain, ytest = split_data(data, x)
    current_step += step_size
    status_text.text("{:.0f}% Complete".format(current_step))
    progress_bar.progress(current_step / 100)
    update_time(time_display, start_time, time.time())

    # Xây dựng mô hình logistic regression
    start_time = time.time()
    build_model.logistic_regression(xtrain, xtest, ytrain, ytest)
    current_step += step_size
    status_text.text("{:.0f}% Complete".format(current_step))
    progress_bar.progress(current_step / 100)
    update_time(time_display, start_time, time.time())


def read_the_results_logistic():
    with open(PATH_SAVE_TEST_AUC + 'fb-pages-food_auc_record_logistic.txt', 'r') as f:
        data = f.readlines()
    for line in data:
        line = line.strip()
        if line:
            auc, time = line.split()
            auc_value = float(auc.split(":")[1])
            annotations_auc_logis = float(time.split(":")[1])
    return auc_value, annotations_auc_logis
