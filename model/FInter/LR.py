import tensorflow as tf

from model.BaseModel import BaseModel


class LR(BaseModel):
    def __init__(self, statical_dict, emb_dim, seq_max_len, user_f_num, use_neg_seq, **params):
        super(LR, self).__init__(statical_dict, emb_dim, seq_max_len, user_f_num, use_neg_seq)

        inp = tf.concat([self.user_emb, self.item_emb, self.item_his_emb_sum], 1)
        dnn1 = tf.layers.dense(inp, 2, activation=None, name='f1')
        self.y_pred = tf.nn.softmax(dnn1)
        self.build_logloss()
