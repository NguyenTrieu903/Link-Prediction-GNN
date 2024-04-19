from SEAL.model import gnn
import numpy as np
from SEAL.utils import utils
from SEAL.config import data, subgraph
import streamlit as st
from sklearn import metrics
import time
from constant import *
import re
import pandas as pd


def execute(is_directed, test_ratio, dimension, hop, learning_rate, top_k=60, epoch=10):
    st.title("Seal Framework")
    positive, negative, nodes_size = data.load_data(is_directed)
    embedding_feature = data.learning_embedding(positive, negative, nodes_size, test_ratio, dimension, is_directed)
    graphs_adj, labels, vertex_tags, node_size_list, sub_graphs_nodes, tags_size = \
        subgraph.link2subgraph(positive, negative, nodes_size, test_ratio, hop, is_directed)

    D_inverse, A_tilde, Y, X, nodes_size_list, initial_feature_dimension = data.create_input_for_gnn_fly(
        graphs_adj, labels, vertex_tags, node_size_list, sub_graphs_nodes, embedding_feature, None, tags_size)
    D_inverse_train, D_inverse_test, A_tilde_train, A_tilde_test, X_train, X_test, Y_train, Y_test, \
        nodes_size_list_train, nodes_size_list_test = utils.split_train_test(D_inverse, A_tilde, X, Y, nodes_size_list)

    model = gnn.build_model(top_k, initial_feature_dimension, nodes_size_list_train, nodes_size_list_test,
                            learning_rate, debug=False)
    start_t = time.time()
    gnn.train(model, X_train, D_inverse_train, A_tilde_train, Y_train, nodes_size_list_train, epoch)
    #prediction = gnn.predict(model, X_test[0], Y_test[0], A_tilde_test[0], D_inverse_test[0], nodes_size_list_test[0])
    test_acc, prediction_one, prediction, pos_scores, pos_score_value_one = gnn.predict(model, X_test, Y_test, A_tilde_test, D_inverse_test, nodes_size_list_test, debug=False)
    end_t = time.time()

    # Tinh AUC
    auc = metrics.roc_auc_score(y_true=np.squeeze(Y_test), y_score=np.squeeze(pos_scores))
    
    st.write("AUC: %f" % auc)
    #print("Probability for prediction is: ", prediction_one[0])
    st.write("The predicted probability for the first element is: ", prediction_one[0])
    st.write("Test acc:",test_acc)
    st.write("Time consumption: ", end_t - start_t)

    with open(PATH_SAVE_TEST_AUC + 'fb-pages-food_auc_record_seal.txt', 'w') as f:
            f.write('AUC:' + str(round(auc, 4)) + '   ' + 'Time:' + 
                    str(round(end_t - start_t, 4)) + '   ' + 'Test_acc:' + str(round(test_acc, 4)) + '\n')
            f.write('pos_score_value_one:' + ', '.join(map(str, pos_score_value_one)) + '\n')
            f.write('prediction_one:' + ', '.join(map(str, prediction_one[0])) + '\n')

def read_the_results_seal():
    time_value = None
    test_acc_value = None
    auc_value = None
    pos_score_value_one = None
    prediction_one = None
    with open(PATH_SAVE_TEST_AUC + 'fb-pages-food_auc_record_seal.txt', 'r') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            line = lines[i].strip()
            if i == 0:
                AUC, time, test = line.split()
                auc_value = float(AUC.split(":")[1])
                time_value = float(time.split(":")[1])
                test_acc_value = float(test.split(":")[1])
                
            elif i == 1:
                tmp = line.split(":")[1]
                numbers = re.findall(r"[-+]?\d*\.\d+|\d+", tmp)
                value_one = float(numbers[0])
                value_two = float(numbers[1])
                pos_score_value_one = pd.DataFrame({0: [value_one], 1: [value_two]})
                
            elif i == 2:
                prediction_one = int(line.split(":")[1])
                

    return auc_value, time_value, test_acc_value, pos_score_value_one, prediction_one