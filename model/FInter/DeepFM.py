import tensorflow as tf

from model.BaseModel import BaseModel
from layer.interaction import FM
from layer.utils import prelu


class DeepFM(BaseModel):
    def __init__(self, statical_dict, emb_dim, seq_max_len, user_f_num, use_neg_seq, **params):
        super(DeepFM, self).__init__(statical_dict, emb_dim, seq_max_len, user_f_num, use_neg_seq)

        inp2 = tf.concat(
            [self.target_user, tf.expand_dims(self.target_item, axis=1), tf.expand_dims(self.target_cate, axis=1)], 1)

        self.fm = FM()(inp2)
        fm_wide = tf.layers.dense(self.fm, 2, activation=None, name='fm')

        inp = tf.concat([self.user_emb, self.item_emb, self.item_his_emb_sum], 1)
        # Fully connected layer
        bn1 = tf.layers.batch_normalization(inputs=inp, name='bn1')
        dnn1 = tf.layers.dense(bn1, 200, activation=None, name='f1')
        dnn1 = prelu(dnn1, 'p1')
        dnn2 = tf.layers.dense(dnn1, 80, activation=None, name='f2')
        dnn2 = prelu(dnn2, 'p2')
        dnn3 = tf.layers.dense(dnn2, 2, activation=None, name='f3')

        self.y_pred = tf.nn.softmax(dnn3 + fm_wide)
        self.build_logloss()
