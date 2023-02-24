import tensorflow as tf

from model.BaseModel import BaseModel


class DNN(BaseModel):
    def __init__(self, statical_dict, emb_dim, seq_max_len, user_f_num, use_neg_seq, **params):
        super(DNN, self).__init__(statical_dict, emb_dim, seq_max_len, user_f_num, use_neg_seq)

        inp = tf.concat([self.user_emb, self.item_emb, self.item_his_emb_sum], 1)

        self.build_fc_net(inp, use_dice=False)
        self.build_logloss()
