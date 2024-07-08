from SEAL.model import gnn
from assets.theme import update_time, plot_auc
import numpy as np
from SEAL.utils import utils
from SEAL.config import data, subgraph
import streamlit as st
from sklearn import metrics
import time
from constant import *
import re
import pandas as pd
import matplotlib.pyplot as plt
import optuna


def execute(is_directed, test_ratio, dimension, hop, learning_rate, top_k=60, epoch=10):

    #TẠO THANH PROGRESS CHO CHƯƠNG TRÌNH TRONG GIAO DIỆN
    global seconds_passed
    seconds_passed = 0 
    total_steps = 8
    step_size = 100 / total_steps
    current_step = 0

    progress_bar = st.sidebar.progress(current_step)
    status_text = st.sidebar.empty()
    time_display = st.sidebar.empty()
    status_text.text("0% Complete")

    # TẢI DỮ LIỆU TỪ FILE VÀO CHƯƠNG TRÌNH
    start_time = time.time()
    positive, negative, nodes_size = data.load_data(is_directed)
    current_step += step_size
    status_text.text("{:.0f}% Complete".format(current_step))
    progress_bar.progress(current_step / 100) 
    update_time(time_display, start_time, time.time())

    # NHÚNG DỮ LIỆU TẠO THÀNH CÁC EMBEDDING
    embedding_feature = data.learning_embedding(positive, negative, nodes_size, test_ratio, dimension, is_directed)
    current_step += step_size
    status_text.text("{:.0f}% Complete".format(current_step))
    progress_bar.progress(current_step / 100)  
    update_time(time_display, start_time, time.time())

    # CHIA DỮ LIỆU THÀNH CÁC TIỂU ĐỒ THỊ
    start_time = time.time()
    graphs_adj, labels, vertex_tags, node_size_list, sub_graphs_nodes, tags_size = \
        subgraph.link2subgraph(positive, negative, nodes_size, test_ratio, hop, is_directed)
    current_step += step_size
    status_text.text("{:.0f}% Complete".format(current_step))
    progress_bar.progress(current_step / 100)  
    update_time(time_display, start_time, time.time())

    # TẠO DỮ LIỆU ĐẦU VÀO CHO MÔ HÌNH
    start_time = time.time()
    D_inverse, A_tilde, Y, X, nodes_size_list, initial_feature_dimension = data.create_input_for_gnn_fly(
        graphs_adj, labels, vertex_tags, node_size_list, sub_graphs_nodes, embedding_feature, None, tags_size)
    current_step += step_size
    status_text.text("{:.0f}% Complete".format(current_step))
    progress_bar.progress(current_step / 100)  
    update_time(time_display, start_time, time.time())

    # CHIA DỮ LIỆU THÀNH TẬP TRAIN VÀ TẬP TEST
    start_time = time.time()
    D_inverse_train, D_inverse_test, A_tilde_train, A_tilde_test, X_train, X_test, Y_train, Y_test, \
        nodes_size_list_train, nodes_size_list_test = utils.split_train_test(D_inverse, A_tilde, X, Y, nodes_size_list)
    current_step += step_size
    status_text.text("{:.0f}% Complete".format(current_step))
    progress_bar.progress(current_step / 100)  
    update_time(time_display, start_time, time.time())

    # XÂY DỰNG MÔ HÌNH
    start_time = time.time()
    model = gnn.build_model(top_k, initial_feature_dimension, nodes_size_list_train, nodes_size_list_test,
                            learning_rate, debug=False)
    current_step += step_size
    status_text.text("{:.0f}% Complete".format(current_step))
    progress_bar.progress(current_step / 100)  
    update_time(time_display, start_time, time.time())

    # HUẤN LUYỆN MÔ HÌNH
    start_t = time.time()
    gnn.train(model, X_train, D_inverse_train, A_tilde_train, Y_train, nodes_size_list_train, epoch)
    end_t = time.time()
    current_step += step_size
    status_text.text("{:.0f}% Complete".format(current_step))
    progress_bar.progress(current_step / 100)  
    update_time(time_display, start_t, time.time())

    # DỰ ĐOÁN KẾT QUẢ
    start_time = time.time()
    test_acc, prediction, pos_scores = gnn.predict(model, X_test, Y_test, A_tilde_test, D_inverse_test, nodes_size_list_test)
    # Tinh AUC
    # y_true là nhãn thực tế của các mẫu dữ liệu trong tập kiểm tra 
    #  pos_scores là kết quá dự đoán của mô hình.
    auc = metrics.roc_auc_score(y_true=np.squeeze(Y_test), y_score=np.squeeze(pos_scores))
    plot_auc(ytest=np.squeeze(Y_test), predictions=np.squeeze(pos_scores), roc = auc, name = "roc_curve_seal.png")
    current_step += step_size
    status_text.text("{:.0f}% Complete".format(current_step))
    progress_bar.progress(current_step / 100)  
    update_time(time_display, start_time, time.time())
    
    st.write("Time consumption: ", end_t - start_t)

    # GHI KẾT QUẢ VÀO FILE 
    with open(PATH_SAVE_TEST_AUC + 'fb-pages-food_auc_record_seal.txt', 'w') as f:
            f.write('AUC:' + str(round(auc, 4)) + '   ' + 'Time:' + 
                    str(round(end_t - start_t, 4)) + '\n')


# ĐỌC KẾT QUẢ TỪ FILE
def read_the_results_seal():
    time_value = None
    auc_value = None
    with open(PATH_SAVE_TEST_AUC + 'fb-pages-food_auc_record_seal.txt', 'r') as f:
        line = f.read()
        AUC, time = line.split()
        auc_value = float(AUC.split(":")[1])
        time_value = float(time.split(":")[1])
                

    return auc_value, time_value