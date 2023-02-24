import numpy as np
import tensorflow as tf

from model.BaseModel import BaseModel
from layer.attention import attention_v1


class DIN(BaseModel):
    def __init__(self, statical_dict, emb_dim, seq_max_len, user_f_num, use_neg_seq, **params):
        super(DIN, self).__init__(statical_dict, emb_dim, seq_max_len, user_f_num, use_neg_seq)

        # 为了保留用户兴趣强度，没有进行softmax归一化
        attention_output = attention_v1(self.item_emb, self.item_his_emb, self.item_mask_ph, mode='SUM', softmax_stag=1,
                                        stag='din_stag', return_alphas=False)
        user_behavior_rep = tf.reduce_sum(attention_output, 1)

        # 拼接，[None,EI]  [None,EU], [None,EI]
        inp = tf.concat([user_behavior_rep, self.user_emb, self.item_his_emb_sum, self.item_emb,
                         self.item_emb * self.item_his_emb_sum], axis=1)

        self.build_fc_net(inp, use_dice=True)
        self.build_logloss()
