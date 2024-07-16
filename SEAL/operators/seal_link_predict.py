import os
import time

import tensorflow as tf

from SEAL.config import data, subgraph
from SEAL.model import gnn
from SEAL.utils import utils
from constant import *


def execute(is_directed, test_ratio, dimension, hop, learning_rate, top_k=60, epoch=10):
    # TẢI DỮ LIỆU TỪ FILE VÀO CHƯƠNG TRÌNH
    positive, negative, nodes_size = data.load_data(is_directed)

    # NHÚNG DỮ LIỆU TẠO THÀNH CÁC EMBEDDING
    embedding_feature = data.learning_embedding(positive, negative, nodes_size, test_ratio, dimension, is_directed)
    graphs_adj, labels, vertex_tags, node_size_list, sub_graphs_nodes, tags_size = \
        subgraph.link2subgraph(positive, negative, nodes_size, test_ratio, hop, is_directed)

    # TẠO DỮ LIỆU ĐẦU VÀO CHO MÔ HÌNH
    D_inverse, A_tilde, Y, X, nodes_size_list, initial_feature_dimension = data.create_input_for_gnn_fly(
        graphs_adj, labels, vertex_tags, node_size_list, sub_graphs_nodes, embedding_feature, None, tags_size)

    # CHIA DỮ LIỆU THÀNH TẬP TRAIN VÀ TẬP TEST
    D_inverse_train, D_inverse_test, A_tilde_train, A_tilde_test, X_train, X_test, Y_train, Y_test, \
        nodes_size_list_train, nodes_size_list_test = utils.split_train_test(D_inverse, A_tilde, X, Y, nodes_size_list)

    # XÂY DỰNG MÔ HÌNH
    start_time = time.time()
    model = gnn.build_model(top_k, initial_feature_dimension, nodes_size_list_train, nodes_size_list_test,
                            learning_rate, debug=False)

    # HUẤN LUYỆN MÔ HÌNH
    # start_t = time.time()
    # gnn.train(model, X_train, D_inverse_train, A_tilde_train, Y_train, nodes_size_list_train, epoch)
    # end_t = time.time()

    # DỰ ĐOÁN KẾT QUẢ
    start_time = time.time()
    test_acc, prediction, pos_scores, pre_y_value, pos_score_value = gnn.predict(model, X_test, Y_test, A_tilde_test, D_inverse_test, nodes_size_list_test)
    print("Probability for the 420th X_test ", pos_score_value)
    print("X_test[420] ", X_test[420])
    # Tinh AUC
    # y_true là nhãn thực tế của các mẫu dữ liệu trong tập kiểm tra 
    #  pos_scores là kết quá dự đoán của mô hình.
    # auc = metrics.roc_auc_score(y_true=np.squeeze(Y_test), y_score=np.squeeze(pos_scores))

    # st.write("Time consumption: ", end_t - start_t)
    #
    # # GHI KẾT QUẢ VÀO FILE
    # with open(PATH_SAVE_TEST_AUC + 'fb-pages-food_auc_record_seal.txt', 'w') as f:
    #         f.write('AUC:' + str(round(auc, 4)) + '   ' + 'Time:' +
    #                 str(round(end_t - start_t, 4)) + '\n')


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


if __name__ == '__main__':
    execute(0, 0.1, 100, "auto", 0.00001)
    # with tf.Session() as sess:
    #     saver = tf.train.import_meta_graph('/home/nhattrieu-machine/Documents/2WL_link_pred-main/SEAL/model/model-1000.meta')
    #     saver.restore(sess, tf.train.latest_checkpoint('/home/nhattrieu-machine/Documents/2WL_link_pred-main/SEAL/model'))
    #     graph = tf.get_default_graph()
    #     for var in tf.compat.v1.global_variables():
    #         print(f"Variable name: {var.name}")
    #         print(sess.run(var))
    # tf.compat.v1.disable_eager_execution()
    #
    # # File path to the meta file and checkpoint
    # meta_file = '/home/nhattrieu-machine/Documents/2WL_link_pred-main/SEAL/model/model-1000.meta'
    # checkpoint = '/home/nhattrieu-machine/Documents/2WL_link_pred-main/SEAL/model'
    #
    # # Start a session
    # with tf.compat.v1.Session() as sess:
    #     # Load the meta graph and weights
    #     saver = tf.compat.v1.train.import_meta_graph(meta_file)
    #     saver.restore(sess, checkpoint)
    #
    #     # Access the graph
    #     graph = tf.compat.v1.get_default_graph()
    #
    #     # Print all variables in the graph
    #     for var in tf.compat.v1.global_variables():
    #         print(f"Variable name: {var.name}")
    #         print(sess.run(var))
