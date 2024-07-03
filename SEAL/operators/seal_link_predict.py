import time

import numpy as np
import streamlit as st
from sklearn import metrics

from SEAL.config import data, subgraph
from SEAL.model import gnn
from SEAL.utils import utils
from assets.theme import update_time, plot_auc
from constant import *


def execute(is_directed, test_ratio, dimension, hop, learning_rate, top_k=60, epoch=10):
    global seconds_passed
    seconds_passed = 0
    total_steps = 8
    step_size = 100 / total_steps
    current_step = 0

    progress_bar = st.sidebar.progress(current_step)
    status_text = st.sidebar.empty()
    time_display = st.sidebar.empty()
    status_text.text("0% Complete")

    # Load dữ liệu tu file
    start_time = time.time()
    positive, negative, nodes_size = data.load_data(is_directed)
    current_step += step_size
    status_text.text("{:.0f}% Complete".format(current_step))
    progress_bar.progress(current_step / 100)
    update_time(time_display, start_time, time.time())

    # Embedding operators
    embedding_feature = data.learning_embedding(positive, negative, nodes_size, test_ratio, dimension, is_directed)
    current_step += step_size
    status_text.text("{:.0f}% Complete".format(current_step))
    progress_bar.progress(current_step / 100)
    update_time(time_display, start_time, time.time())

    # Tạo tiểu đồ thị 
    start_time = time.time()
    graphs_adj, labels, vertex_tags, node_size_list, sub_graphs_nodes, tags_size = \
        subgraph.link2subgraph(positive, negative, nodes_size, test_ratio, hop, is_directed)
    current_step += step_size
    status_text.text("{:.0f}% Complete".format(current_step))
    progress_bar.progress(current_step / 100)
    update_time(time_display, start_time, time.time())

    # Tạo dữ liệu đâu vào cho mô hình
    start_time = time.time()
    D_inverse, A_tilde, Y, X, nodes_size_list, initial_feature_dimension = data.create_input_for_gnn_fly(
        graphs_adj, labels, vertex_tags, node_size_list, sub_graphs_nodes, embedding_feature, None, tags_size)
    current_step += step_size
    status_text.text("{:.0f}% Complete".format(current_step))
    progress_bar.progress(current_step / 100)
    update_time(time_display, start_time, time.time())

    # Chia dữ liệu thành tập train và tập test
    start_time = time.time()
    D_inverse_train, D_inverse_test, A_tilde_train, A_tilde_test, X_train, X_test, Y_train, Y_test, \
        nodes_size_list_train, nodes_size_list_test = utils.split_train_test(D_inverse, A_tilde, X, Y, nodes_size_list)
    current_step += step_size
    status_text.text("{:.0f}% Complete".format(current_step))
    progress_bar.progress(current_step / 100)
    update_time(time_display, start_time, time.time())

    # Xây dựng mô hình
    start_time = time.time()
    model = gnn.build_model(top_k, initial_feature_dimension, nodes_size_list_train, nodes_size_list_test,
                            learning_rate, debug=False)
    current_step += step_size
    status_text.text("{:.0f}% Complete".format(current_step))
    progress_bar.progress(current_step / 100)
    update_time(time_display, start_time, time.time())

    # Huấn luyện mô hình
    start_t = time.time()
    gnn.train(model, X_train, D_inverse_train, A_tilde_train, Y_train, nodes_size_list_train, epoch)
    end_t = time.time()
    current_step += step_size
    status_text.text("{:.0f}% Complete".format(current_step))
    progress_bar.progress(current_step / 100)
    update_time(time_display, start_t, time.time())

    # Dự đoán kết quả
    start_time = time.time()
    test_acc, prediction, pos_scores = gnn.predict(model, X_test, Y_test, A_tilde_test, D_inverse_test,
                                                   nodes_size_list_test)
    # Tinh AUC
    auc = metrics.roc_auc_score(y_true=np.squeeze(Y_test), y_score=np.squeeze(pos_scores))
    plot_auc(ytest=np.squeeze(Y_test), predictions=np.squeeze(pos_scores), roc=auc, name="roc_curve_seal.png")
    current_step += step_size
    status_text.text("{:.0f}% Complete".format(current_step))
    progress_bar.progress(current_step / 100)
    update_time(time_display, start_time, time.time())

    st.write("Time consumption: ", end_t - start_t)

    # Ghi kết quả vào file
    with open(PATH_SAVE_TEST_AUC + 'fb-pages-food_auc_record_seal.txt', 'w') as f:
        f.write('AUC:' + str(round(auc, 4)) + '   ' + 'Time:' +
                str(round(end_t - start_t, 4)) + '\n')


def read_the_results_seal():
    time_value = None
    auc_value = None
    with open(PATH_SAVE_TEST_AUC + 'fb-pages-food_auc_record_seal.txt', 'r') as f:
        line = f.read()
        AUC, time = line.split()
        auc_value = float(AUC.split(":")[1])
        time_value = float(time.split(":")[1])
    return auc_value, time_value
