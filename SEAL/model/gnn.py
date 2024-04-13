import numpy as np
import tensorflow as tf
from tqdm import tqdm
from sklearn import metrics

import constant
from SEAL.utils import modelGCN

# import modelGCN


GRAPH_CONV_LAYER_CHANNEL = 32
CONV1D_1_OUTPUT = 16
CONV1D_2_OUTPUT = 32
CONV1D_1_FILTER_WIDTH = GRAPH_CONV_LAYER_CHANNEL * 3
CONV1D_2_FILTER_WIDTH = 5
DENSE_NODES = 128
DROP_OUTPUT_RATE = 0.5
LEARNING_RATE_BASE = 0.00004
LEARNING_RATE_DECAY = 0.99

def build_model(top_k, initial_channels, nodes_size_list_train, nodes_size_list_test, learning_rate, debug):
    D_inverse_pl = tf.placeholder(dtype=tf.float32, shape=[None, None], name="D_inverse_pl")
    A_tilde_pl = tf.placeholder(dtype=tf.float32, shape=[None, None], name="A_tilde_pl")
    X_pl = tf.placeholder(dtype=tf.float32, shape=[None, initial_channels], name="X_pl")
    Y_pl = tf.placeholder(dtype=tf.int32, shape=[1], name="Y-placeholder")
    node_size_pl = tf.placeholder(dtype=tf.int32, shape=[], name="node-size-placeholder")
    is_train = tf.placeholder(dtype=tf.uint8, shape=[], name="is-train-or-test")

    # trainable parameters of graph convolution layer
    # tao cac bien co ten la graph_weight_1 voi cac gia tri ngau nhien cu the.Duoc su dung trong qua trinh khoi tao cac trong so cua mang no-ron 
    graph_weight_1 = tf.Variable(tf.truncated_normal(shape=[initial_channels, GRAPH_CONV_LAYER_CHANNEL], stddev=0.1, dtype=tf.float32), name="graph_weight_1")
    graph_weight_2 = tf.Variable(tf.truncated_normal(shape=[GRAPH_CONV_LAYER_CHANNEL, GRAPH_CONV_LAYER_CHANNEL], stddev=0.1, dtype=tf.float32), name="graph_weight_2")
    graph_weight_3 = tf.Variable(tf.truncated_normal(shape=[GRAPH_CONV_LAYER_CHANNEL, GRAPH_CONV_LAYER_CHANNEL], stddev=0.1, dtype=tf.float32), name="graph_weight_3")
    graph_weight_4 = tf.Variable(tf.truncated_normal(shape=[GRAPH_CONV_LAYER_CHANNEL, 1], stddev=0.1, dtype=tf.float32), name="graph_weight_4")

    # GRAPH CONVOLUTION LAYER
    # thuc hien mot loat cac phep tinh ma tran de (nhu la 1 lop xu ly) trong mang no-ron.
    gl_1_XxW = tf.matmul(X_pl, graph_weight_1)
    gl_1_AxXxW = tf.matmul(A_tilde_pl, gl_1_XxW)
    Z_1 = tf.nn.tanh(tf.matmul(D_inverse_pl, gl_1_AxXxW))

    gl_2_XxW = tf.matmul(Z_1, graph_weight_2)
    gl_2_AxXxW = tf.matmul(A_tilde_pl, gl_2_XxW)
    Z_2 = tf.nn.tanh(tf.matmul(D_inverse_pl, gl_2_AxXxW))

    gl_3_XxW = tf.matmul(Z_2, graph_weight_3)
    gl_3_AxXxW = tf.matmul(A_tilde_pl, gl_3_XxW)
    Z_3 = tf.nn.tanh(tf.matmul(D_inverse_pl, gl_3_AxXxW))

    gl_4_XxW = tf.matmul(Z_3, graph_weight_4)
    gl_4_AxXxW = tf.matmul(A_tilde_pl, gl_4_XxW)
    Z_4 = tf.nn.tanh(tf.matmul(D_inverse_pl, gl_4_AxXxW))

    # de noi cac tensor theo 1 truc cu the. axis = 1 => duoc noi theo chieu ngang
    graph_conv_output = tf.concat([Z_1, Z_2, Z_3], axis=1)  # shape=(node_size/None, 32+32+32)

    # Ham duoc su dung de tinh toan cac thong ke mo ta cho mot bien Tensorflow nhat dinh nhu: mean, variance, max, min. 
    def variable_summary(var):
        var_mean = tf.reduce_mean(var)
        var_variance = tf.square(tf.reduce_mean(tf.square(var - tf.reduce_mean(var))))
        var_max = tf.reduce_max(var)
        var_min = tf.reduce_min(var)
        return var_mean, var_variance, var_max, var_min

    if debug:
        var_mean, var_variance, var_max, var_min = variable_summary(graph_weight_1)

    # SORT POOLING LAYER: Su dung de tinh toan nguong dua tren phan tram (percentile) cua kich thuoc nodes trong mot tap hop cac do thi.
    nodes_size_list = list(nodes_size_list_train) + list(nodes_size_list_test)
    threshold_k = int(np.percentile(nodes_size_list, top_k))
    print("%s%% graphs have nodes less then %s." % (top_k, threshold_k))

    # Dung de lua chon cac phan tu quan trong tu graph_conv_output va tao chung thanh mot tensor moi.  => co the toi uu (thay the cach lay cac phan tu quan trong: Z_4[:, 0])
    graph_conv_output_stored = tf.gather(graph_conv_output, tf.nn.top_k(Z_4[:, 0], node_size_pl).indices)

    # Tao ra 1 tensor moi dua tren dieu kien: node_size_pl < threshold_k
    graph_conv_output_top_k = tf.cond(tf.less(node_size_pl, threshold_k),
                                      lambda: tf.concat(axis=0,
                                                        values=[graph_conv_output_stored,
                                                                tf.zeros(dtype=tf.float32,
                                                                         shape=[threshold_k-node_size_pl,
                                                                                GRAPH_CONV_LAYER_CHANNEL*3])]),
                                      lambda: tf.slice(graph_conv_output_stored, begin=[0, 0], size=[threshold_k, -1]))

    # FLATTEN LAYER
    # Su dung de bien doi tensor graph_conv_output_top_k thanh mot tensor graph_conv_output_flatten co hinh dang moi
    graph_conv_output_flatten = tf.reshape(graph_conv_output_top_k, shape=[1, GRAPH_CONV_LAYER_CHANNEL*3*threshold_k, 1])
    assert graph_conv_output_flatten.shape == [1, GRAPH_CONV_LAYER_CHANNEL*3*threshold_k, 1]

    # 1-D CONVOLUTION LAYER 1:
    # kernel = (filter_width, in_channel, out_channel)
    # Thuc hien phep tinh chap 1 chieu tren du lieu dau vao bang cach su dung bo lap conv1d_kernel_1
    # Tao mot bien conv1d_kernel_1 de luu tru bo loc cho phep tich chap mot chieu
    conv1d_kernel_1 = tf.Variable(tf.truncated_normal(shape=[CONV1D_1_FILTER_WIDTH, 1, CONV1D_1_OUTPUT], stddev=0.1, dtype=tf.float32))
    # Su dung tf.nn.conv1d de thuc hien phep tich chap 1 chieu
    conv_1d_a = tf.nn.conv1d(graph_conv_output_flatten, conv1d_kernel_1, stride=CONV1D_1_FILTER_WIDTH, padding="VALID")
    assert conv_1d_a.shape == [1, threshold_k, CONV1D_1_OUTPUT]

    # 1-D CONVOLUTION LAYER 2:
    conv1d_kernel_2 = tf.Variable(tf.truncated_normal(shape=[CONV1D_2_FILTER_WIDTH, CONV1D_1_OUTPUT, CONV1D_2_OUTPUT], stddev=0.1, dtype=tf.float32))
    conv_1d_b = tf.nn.conv1d(conv_1d_a, conv1d_kernel_2, stride=1, padding="VALID")
    assert conv_1d_b.shape == [1, threshold_k - CONV1D_2_FILTER_WIDTH + 1, CONV1D_2_OUTPUT]
    # Su dung de lam phang (flatten) dau ra cua mot lop convolutional 1D thanh mot vecto 
    conv_output_flatten = tf.layers.flatten(conv_1d_b)

    # Sau khi thuc hien cac phep tinh tren ma tran dau vao -> trich xuat dac trung thong qua cac convolution layer, sau do dua vao cac lop neural network de thuc hien du doan
    # DENSE LAYER: Dung de tao ra mot lop ket noi day du (fully connected layer) trong mang noron.
    # Dung de tao ma tran trong so. Kich thuoc ma tran trong so la: [so luong dac trung dau vao, so do vi trong lop ket noi day du]
    # Tao lop neural network dau tien
    weight_1 = tf.Variable(tf.truncated_normal(shape=[int(conv_output_flatten.shape[1]), DENSE_NODES], stddev=0.1), name="weight_1")
    # Khoi tao bias cua lop ket noi day du. 
    bias_1 = tf.Variable(tf.zeros(shape=[DENSE_NODES]), name="bias_1")
    # Dung de thuc hien cac phep tinh de tao dau ra cua mot lop ket noi day du trong mang no-ron su dung ham kich hoat RELU
    dense_z = tf.nn.relu(tf.matmul(conv_output_flatten, weight_1) + bias_1)
    # Thuc hien qua trinh loai bo ngau nhien 1 phan cua dau ra tu lop ket noi day du -> giup ngan chan viec mang no-ron hoc thuoc qua muc va giam nguy co onverfitting
    if is_train == 1:
        dense_z = tf.layers.dropout(dense_z, DROP_OUTPUT_RATE)

    # Tao lop neural network thu 2
    weight_2 = tf.Variable(tf.truncated_normal(shape=[DENSE_NODES, 2]), name="weight_2")
    bias_2 = tf.Variable(tf.zeros(shape=[2]), name="bias_2")
    pre_y = tf.matmul(dense_z, weight_2) + bias_2
    # Sau do ap dung ham tinh sac xuat softmax tren dau ra cua lop thu 2
    pos_score = tf.nn.softmax(pre_y)

    # Duoc su dung de tinh toan ham mat mat (loss) cho mo hinh phan loai su dung softmax -> co the toi uu bang viec su dung cac ham sac xuat khac
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=Y_pl, logits=pre_y))
    global_step = tf.Variable(0, trainable=False)
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step)
    # return D_inverse_pl, A_tilde_pl, X_pl, Y_pl, node_size_pl, is_train, pos_score, loss, global_step, pre_y
    return modelGCN.model_GCN(D_inverse_pl, A_tilde_pl, X_pl, Y_pl, node_size_pl, is_train, pre_y, pos_score, train_op, global_step, loss)

def train(model, X_train, D_inverse_train, A_tilde_train, Y_train, nodes_size_list_train, epoch):
    D_inverse_pl = model.D_inverse_pl
    A_tilde_pl = model.A_tilde_pl
    X_pl = model.X_pl
    Y_pl = model.Y_pl
    node_size_pl = model.node_size_pl
    is_train = model.is_train
    pre_y = model.pre_y
    train_op, global_step, loss = model.train_op, model.global_step, model.loss

    train_data_size = X_train.shape[0]
    print("train_data_size: ", train_data_size)
    with tf.Session() as sess:
        saver = tf.train.Saver()
        for epoch in range(epoch):
            print("start training gnn.")
            sess.run(tf.global_variables_initializer())
            batch_index = 0
            for _ in tqdm(range(train_data_size)):
                batch_index = batch_index + 1 if batch_index < train_data_size - 1 else 0
                feed_dict = {D_inverse_pl: D_inverse_train[batch_index],
                             A_tilde_pl: A_tilde_train[batch_index],
                             X_pl: X_train[batch_index],
                             Y_pl: Y_train[batch_index],
                             node_size_pl: nodes_size_list_train[batch_index],
                             is_train: 1
                             }
                loss_value, _, _ = sess.run([loss, train_op, global_step], feed_dict=feed_dict)

            train_acc = 0
            for i in tqdm(range(train_data_size)):
                    feed_dict = {D_inverse_pl: D_inverse_train[i], A_tilde_pl: A_tilde_train[i],
                                X_pl: X_train[i], Y_pl: Y_train[i], node_size_pl: nodes_size_list_train[i], is_train: 0}
                    pre_y_value = sess.run(pre_y, feed_dict=feed_dict)
                    if np.argmax(pre_y_value, 1) == Y_train[i]:
                        train_acc += 1
            train_acc = train_acc / train_data_size    
            print("After %5s epoch, training acc %f, the loss is %f." % (epoch, train_acc, loss_value))    
        saver.save(sess, constant.MODEL_SAVE_PATH ,global_step=1000)

def predict(model, X_test, A_tilde_test, D_inverse_test, nodes_size_list_test):
    pre_y = model.pre_y
    # Y_pl = model.Y_pl
    pos_score = model.pos_score

    test_acc, prediction, scores = 0, [], []

    with tf.Session() as sess:
        # load model
        sess.run(tf.global_variables_initializer())
        saver = tf.train.import_meta_graph(constant.MODEL_READ_PATH)
        saver.restore(sess,tf.train.latest_checkpoint(constant.CHECKPOINT_MODEL))
        graph = tf.get_default_graph()

        # get paramaters
        X_pl = graph.get_tensor_by_name("X_pl:0")
        A_tilde_pl = graph.get_tensor_by_name("A_tilde_pl:0")
        D_inverse_pl = graph.get_tensor_by_name("D_inverse_pl:0")
        is_train = graph.get_tensor_by_name("is-train-or-test:0")
        node_size_pl = graph.get_tensor_by_name("node-size-placeholder:0")
        weight_1 = graph.get_tensor_by_name("weight_1:0")
        weight_2 = graph.get_tensor_by_name("weight_2:0")
        bias_1 = graph.get_tensor_by_name("bias_1:0")
        bias_2 = graph.get_tensor_by_name("bias_2:0")

        graph_weight_1 = graph.get_tensor_by_name("graph_weight_1:0")
        graph_weight_2 = graph.get_tensor_by_name("graph_weight_2:0")
        graph_weight_3 = graph.get_tensor_by_name("graph_weight_3:0")
        graph_weight_4 = graph.get_tensor_by_name("graph_weight_4:0")

        weight_1_value, weight_2_value, bias_1_value, bias_2_value = sess.run([weight_1, weight_2, bias_1, bias_2])
        graph_weight_1_value, graph_weight_2_value, graph_weight_3_value, graph_weight_4_value = sess.run([graph_weight_1, graph_weight_2, graph_weight_3, graph_weight_4])

        
        # saver = tf.train.Saver()
        # saver.restore(sess, '/home/nhattrieu-machine/Documents/SEAL-for-link-prediction-master/model-1000.meta')  # Đường dẫn tới tệp tin đã lưu
        print("Model restored.")
        feed_dict = {X_pl: X_test, is_train: 0, A_tilde_pl: A_tilde_test, D_inverse_pl: D_inverse_test,  node_size_pl: nodes_size_list_test,
                         weight_1:weight_1_value, weight_2:weight_2_value, bias_1:bias_1_value, bias_2:bias_2_value, graph_weight_1:graph_weight_1_value, 
                         graph_weight_2: graph_weight_2_value, graph_weight_3 :graph_weight_3_value, graph_weight_4:graph_weight_4_value
                         }
        pre_y_value, pos_score_value = sess.run([pre_y, pos_score], feed_dict=feed_dict)
        print("pos_score_value: ", pos_score_value)
        prediction.append(np.argmax(pos_score_value, 1))
    return prediction
