import os
import tensorflow as tf

class model_GCN:
  def __init__(self, D_inverse_pl, A_tilde_pl, X_pl, Y_pl, node_size_pl, is_train, pre_y, pos_score, train_op, global_step, loss):
    self.D_inverse_pl = D_inverse_pl
    self.A_tilde_pl = A_tilde_pl
    self.X_pl = X_pl
    self.Y_pl = Y_pl
    self.node_size_pl = node_size_pl
    self.is_train = is_train
    self.pre_y = pre_y
    self.pos_score = pos_score
    self.train_op = train_op
    self.global_step = global_step
    self.loss = loss