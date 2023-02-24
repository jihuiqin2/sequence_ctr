import tensorflow as tf

from model.BaseModel import BaseModel
from layer.interaction import SENETLayer, BilinearInteraction
from layer.utils import prelu


class FiBiNet(BaseModel):
    def __init__(self, statical_dict, emb_dim, seq_max_len, user_f_num, use_neg_seq, **params):
        super(FiBiNet, self).__init__(statical_dict, emb_dim, seq_max_len, user_f_num, use_neg_seq)
        # 整理输入的数据格式
        inp = tf.concat(
            [self.user_emb, self.item_emb, self.item_his_emb_sum,
             self.item_emb * self.item_his_emb_sum], 1)
        target_item = tf.expand_dims(self.target_item, 1)
        target_cate = tf.expand_dims(self.target_cate, 1)
        user_emb = tf.reshape(self.user_emb, [-1, user_f_num, emb_dim])
        input = tf.concat([user_emb, target_item, target_cate], axis=1)  # [b,num,emb]

        # SENETLayer和双线性
        input_list = tf.split(input, user_f_num + 2, axis=1)  # list，num个[b,1,emb]
        senet_emb = SENETLayer()(input_list)
        bilinear_q = BilinearInteraction()(senet_emb)
        bilinear_p = BilinearInteraction()(input_list)
        comb_out = tf.layers.flatten(tf.concat([bilinear_q, bilinear_p], axis=1))

        inp = tf.concat([inp, comb_out], 1)

        # Fully connected layer
        bn1 = tf.layers.batch_normalization(inputs=inp, name='bn1')
        dnn1 = tf.layers.dense(bn1, 200, activation=None, name='f1')
        dnn1 = prelu(dnn1, 'p1')
        dnn2 = tf.layers.dense(dnn1, 80, activation=None, name='f2')
        dnn2 = prelu(dnn2, 'p2')
        dnn3 = tf.layers.dense(dnn2, 2, activation=None, name='f3')
        d_layer_wide = tf.concat([tf.concat([self.item_emb, self.item_his_emb_sum], axis=-1),
                                  self.item_emb * self.item_his_emb_sum], axis=-1)
        d_layer_wide = tf.layers.dense(d_layer_wide, 2, activation=None, name='f_fm')
        self.y_pred = tf.nn.softmax(dnn3 + d_layer_wide)

        self.build_logloss()
