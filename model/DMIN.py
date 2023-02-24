import tensorflow as tf

from model.BaseModel import BaseModel
from layer.attention import self_multi_head_attn_v1, self_multi_head_attn_v2
from layer.attention import multi_attention


class DMIN(BaseModel):
    def __init__(self, statical_dict, emb_dim, seq_max_len, user_f_num, use_neg_seq, head_num, **params):
        super(DMIN, self).__init__(statical_dict, emb_dim, seq_max_len, user_f_num, use_neg_seq)
        # 位置编码
        other_embedding_size = 2
        self.position_his = tf.range(seq_max_len)  # [SL,]
        self.position_embeddings_var = tf.get_variable("position_embeddings_var", [seq_max_len, other_embedding_size])
        self.position_his_emb = tf.nn.embedding_lookup(self.position_embeddings_var, self.position_his)  # [SL,2]
        # 正历史序列，位置编码，在同一纬度上平铺
        self.position_his_emb = tf.tile(self.position_his_emb, [tf.shape(self.item_his_emb)[0], 1])  # [B*SL,2]
        get_pos_shape = self.position_his_emb.get_shape().as_list()[1]
        self.position_his_emb = tf.reshape(self.position_his_emb,
                                           [tf.shape(self.item_his_emb)[0], -1, get_pos_shape])  # [B,SL,2]

        # 多头自注意力
        with tf.name_scope("multi_head_attention1"):
            multi_outputs = self_multi_head_attn_v1(self.item_his_emb, num_dim=emb_dim * 2,
                                                    num_heads=4, dropout_rate=0, is_training=True)
            multi_head_attention_outputs1 = tf.layers.dense(multi_outputs, emb_dim * 4,
                                                            activation=tf.nn.relu)  # [None,SL,4*emb]
            multi_head_attention_outputs1 = tf.layers.dense(multi_head_attention_outputs1, emb_dim * 2)  # [None,SL,EI]
            multi_head_attention_outputs = multi_head_attention_outputs1 + multi_outputs

        # 辅助损失
        if use_neg_seq:
            aux_loss_1 = self.auxiliary_loss(multi_head_attention_outputs[:, :-1, :], self.item_his_emb[:, 1:, :],
                                             self.item_neg_emb[:, 1:, :], self.item_mask_ph[:, 1:], stag="dmin_gru")
            self.aux_loss = aux_loss_1

        inp = tf.concat(
            [self.user_emb, self.item_emb, self.item_his_emb_sum,
             self.item_emb * self.item_his_emb_sum], 1)

        # 多头自注意力2
        with tf.name_scope("multi_head_attention2"):  # emb_dim * 2  (cate和item)
            multi_head_attention2 = self_multi_head_attn_v2(multi_head_attention_outputs, num_dim=emb_dim * 2,
                                                            num_heads=head_num, dropout_rate=0, is_training=True)
            for i, multi_att_v2 in enumerate(multi_head_attention2):
                outputs3 = tf.layers.dense(multi_att_v2, emb_dim * 4, activation=tf.nn.relu)
                outputs3 = tf.layers.dense(outputs3, emb_dim * 2)
                multi_att_v2 = outputs3 + multi_att_v2  # [N,SL,EI]
                with tf.name_scope('Attention_layer' + str(i)):
                    # 这里使用position embedding来算attention [N,1,EI]
                    attention_output, _, _ = multi_attention(self.item_emb, multi_att_v2, self.position_his_emb,
                                                             self.item_mask_ph, stag='dmin_stag_' + str(i))
                    att_fea = tf.reduce_sum(attention_output, 1)  # [N,EI]
                    inp = tf.concat([inp, att_fea], 1)

        # Fully connected layer
        self.build_fc_net(inp, use_dice=True)
        self.build_logloss()
