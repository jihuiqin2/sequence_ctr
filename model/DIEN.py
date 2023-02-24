import tensorflow as tf
from tensorflow.python.keras.layers import Permute, multiply, Dense

from model.BaseModel import BaseModel
from layer.sequence import DynamicGRU
from layer.attention import attention_v1


class DIEN(BaseModel):
    def __init__(self, statical_dict, emb_dim, seq_max_len, user_f_num, use_neg_seq, gru_type, **params):
        super(DIEN, self).__init__(statical_dict, emb_dim, seq_max_len, user_f_num, use_neg_seq)

        # attention RNN layer, emb_dim代表gru隐藏层的维度
        hidden_dim = self.item_his_emb.get_shape().as_list()[-1]
        with tf.name_scope('rnn_1'):
            user_seq_ht, aux_loss = self.interest_extractor(hidden_dim, use_neg_seq, self.item_his_emb,
                                                            self.item_neg_emb, self.user_seq_length_ph)

        with tf.name_scope('rnn_2'):
            # att_weight_normalization归一化
            final_state = self.interest_evolution(hidden_dim, user_seq_ht,
                                                  self.user_seq_length_ph, gru_type=gru_type)
            final_state = tf.reshape(final_state, [-1, final_state.shape[-1]])  # [None,EI]

        inp = tf.concat(
            [final_state, self.user_emb, self.item_his_emb_sum, self.item_emb,
             self.item_emb * self.item_his_emb_sum], axis=1)

        self.build_fc_net(inp, use_dice=True)
        self.build_logloss()

    # 兴趣抽取
    def interest_extractor(self, hidden_dim, use_neg, item_hist_emb, item_neg_emb, user_seq_length_ph):
        user_seq_ht = DynamicGRU(hidden_dim, return_sequence=True, name="gru1")(
            [item_hist_emb, user_seq_length_ph])  # [None,SL,HS] HS=EI
        if use_neg:
            aux_loss = self.auxiliary_loss(user_seq_ht[:, :-1, :], item_hist_emb[:, 1:, :],
                                           item_neg_emb[:, 1:, :],
                                           self.item_mask_ph[:, 1:], stag="gru")
            self.aux_loss = aux_loss

        return user_seq_ht, aux_loss

    # 兴趣演化
    def interest_evolution(self, hidden_dim, user_seq_ht, user_seq_length_ph, gru_type):
        if gru_type not in ["GRU", "AIGRU", "AGRU", "AUGRU"]:
            raise ValueError("gru_type error ")

        # [N,SL,EI] [N,SL]
        _, alphas = attention_v1(self.item_emb, user_seq_ht, self.item_mask_ph,
                                 softmax_stag=1, stag='dien_stag', mode='LIST', return_alphas=True)

        if gru_type == "AIGRU":
            scores = Permute([2, 1])(tf.expand_dims(alphas, -1))  # 维度转换
            hist = multiply([user_seq_ht, Permute([2, 1])(scores)])  # 两个矩阵中对应元素各自相乘
            final_state = DynamicGRU(hidden_dim, gru_type="AIGRU", return_sequence=False, name='AIGRU')(
                [hist, user_seq_length_ph])
        else:
            final_state = DynamicGRU(hidden_dim, gru_type=gru_type, return_sequence=False, name='AGRU')(
                [user_seq_ht, user_seq_length_ph, tf.expand_dims(alphas, -1)])

        return final_state  # [None,1,EI]


"""
z = tf.subtract(x, y)，减法操作，对应位置相减
tf.sequence_mask([1, 3, 2], 5)， 得到3行5列，对应位置为true，其他为false
tf.cast()， 类型转换
tf.tile()：平铺，用于在同一维度上的复制
"""
