import tensorflow as tf

from model.BaseModel import BaseModel
from layer.sequence import ProbAttention, AttentionLayer, PositionEncoding, DynamicGRU
from layer.interaction import InteractionCross, FeatureAFF
from layer.attention import attention_v1


class NewModel(BaseModel):
    def __init__(self, statical_dict, emb_dim, seq_max_len, user_f_num, use_neg_seq, head_num, seed,
                 cross_num=3, l2_reg_cross=1e-5, choose_k=10, **params):
        super(NewModel, self).__init__(statical_dict, emb_dim, seq_max_len, user_f_num, use_neg_seq)

        inp = tf.concat([self.user_emb, self.item_emb, self.item_his_emb_sum, self.item_emb * self.item_his_emb_sum], 1)

        # 【1】增加重要信息的权重
        self.query_pe = PositionEncoding()
        item_his_emb = self.query_pe(self.item_his_emb)  # position embedding
        senet_layer = FeatureAFF(seed=seed)(item_his_emb)

        # 【2】兴趣抽取层
        hidden_dim = self.item_his_emb.get_shape().as_list()[-1]
        att_embedding_size = hidden_dim // head_num
        attn = AttentionLayer(ProbAttention(False, choose_k), 2 * emb_dim, att_embedding_size, head_num)
        dense_gru = attn([senet_layer, senet_layer, senet_layer])

        # 辅助损失
        if use_neg_seq:
            aux_loss_1 = self.auxiliary_loss(dense_gru[:, :-1, :], self.item_his_emb[:, 1:, :],
                                             self.item_neg_emb[:, 1:, :], self.item_mask_ph[:, 1:], stag="new_gru")
            self.aux_loss = aux_loss_1

        # 【3】兴趣更新层
        _, alphas = attention_v1(self.item_emb, dense_gru, self.item_mask_ph,
                                 softmax_stag=1, stag='dien_stag', mode='LIST', return_alphas=True)
        final_state = DynamicGRU(hidden_dim, gru_type='AGRU', return_sequence=False, name='AGRU')(
            [dense_gru, self.user_seq_length_ph, tf.expand_dims(alphas, -1)])
        final_state = tf.reshape(final_state, [-1, final_state.shape[-1]])  # [None,EI]

        # 【4】 分层特征交互层，将所有的特征包括序列进行特征交叉
        # with tf.name_scope("interaction-interaction"):
        #     u_i = tf.concat([self.user_emb, self.item_emb], axis=-1, name='u_i')
        #     u_i = InteractionCross(cross_num, l2_reg=l2_reg_cross)(u_i)

        # 拼接
        inp = tf.concat([inp, final_state], axis=1, name='inp')

        self.build_fc_net(inp)
        self.build_logloss()
