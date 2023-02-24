import copy
import logging

import numpy as np
import math
from math import sqrt
import tensorflow as tf
from tensorflow.python.keras import backend as K

from tensorflow.python.ops.init_ops import Zeros, Ones

from tensorflow.python.ops.init_ops import TruncatedNormal, glorot_uniform_initializer as glorot_uniform, \
    identity_initializer as identity

from tensorflow.python.keras.layers import LSTM, Lambda, Layer, Dropout, Dense
from layer.core import reduce_sum
from layer.contrib.cell_sequence import QAAttGRUCell, VecAttGRUCell
from layer.contrib.rnn_sequence import dynamic_rnn
from layer.utils import reduce_max, reduce_mean, softmax, layer_norm, ProbMask

SELF_ATTN_INF_NEG = -5e4
LOOK_AHEAD_ATTN_INF_NEG = -1e38


# DynamicGRU
class DynamicGRU(Layer):
    def __init__(self, num_units=None, gru_type='GRU', return_sequence=True, **kwargs):
        super(DynamicGRU, self).__init__(**kwargs)
        self.num_units = num_units
        self.return_sequence = return_sequence
        self.gru_type = gru_type

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        input_seq_shape = input_shape[0]
        if self.num_units is None:
            self.num_units = input_seq_shape.as_list()[-1]
        if self.gru_type == "AGRU":
            self.gru_cell = QAAttGRUCell(self.num_units)
        elif self.gru_type == "AUGRU":
            self.gru_cell = VecAttGRUCell(self.num_units)
        else:
            self.gru_cell = tf.nn.rnn_cell.GRUCell(self.num_units)  # AIGRU 和 GRU

        # Be sure to call this somewhere!
        super(DynamicGRU, self).build(input_shape)

    def call(self, input_list):
        if self.gru_type == "GRU" or self.gru_type == "AIGRU":
            rnn_input, sequence_length = input_list
            att_score = None
        else:
            rnn_input, sequence_length, att_score = input_list

        # att_score[N,SL,1]   sequence_length[N,]  rnn_input[N,SL,EI]
        rnn_output, hidden_state = dynamic_rnn(self.gru_cell, inputs=rnn_input, att_scores=att_score,
                                               sequence_length=sequence_length, dtype=tf.float32,
                                               scope=self.name)

        if self.return_sequence:
            return rnn_output  # [None,SL,EI]所有时刻的隐藏层状态
        else:
            return tf.expand_dims(hidden_state, axis=1)  # [None,1,EI]最后一个状态


# BiLSTM
class BiLSTM(Layer):
    def __init__(self, units, layers=2, res_layers=0, dropout_rate=0.2, merge_mode='ave', **kwargs):

        if merge_mode not in ['fw', 'bw', 'sum', 'mul', 'ave', 'concat', None]:
            raise ValueError('Invalid merge mode. '
                             'Merge mode should be one of '
                             '{"fw","bw","sum", "mul", "ave", "concat", None}')

        self.units = units
        self.layers = layers
        self.res_layers = res_layers
        self.dropout_rate = dropout_rate
        self.merge_mode = merge_mode

        super(BiLSTM, self).__init__(**kwargs)
        self.supports_masking = True

    def build(self, input_shape):

        if len(input_shape) != 3:
            raise ValueError(
                "Unexpected inputs dimensions %d, expect to be 3 dimensions" % (len(input_shape)))
        self.fw_lstm = []
        self.bw_lstm = []
        for _ in range(self.layers):
            self.fw_lstm.append(
                LSTM(self.units, dropout=self.dropout_rate, bias_initializer='ones', return_sequences=True,
                     unroll=True))
            self.bw_lstm.append(
                LSTM(self.units, dropout=self.dropout_rate, bias_initializer='ones', return_sequences=True,
                     go_backwards=True, unroll=True))

        super(BiLSTM, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs, **kwargs):

        input_fw = inputs
        input_bw = inputs
        mask = None
        for i in range(self.layers):
            output_fw = self.fw_lstm[i](input_fw)
            output_bw = self.bw_lstm[i](input_bw)
            output_bw = Lambda(lambda x: K.reverse(x, 1), mask=lambda inputs, mask: mask)(output_bw)

            if i >= self.layers - self.res_layers:
                output_fw += input_fw
                output_bw += input_bw
            input_fw = output_fw
            input_bw = output_bw

        output_fw = input_fw
        output_bw = input_bw

        if self.merge_mode == "fw":
            output = output_fw
        elif self.merge_mode == "bw":
            output = output_bw
        elif self.merge_mode == 'concat':
            output = K.concatenate([output_fw, output_bw])
        elif self.merge_mode == 'sum':
            output = output_fw + output_bw
        elif self.merge_mode == 'ave':
            output = (output_fw + output_bw) / 2
        elif self.merge_mode == 'mul':
            output = output_fw * output_bw
        elif self.merge_mode is None:
            output = [output_fw, output_bw]

        return output


# Informer对多头自注意力机制的改进
class ProbAttention(Layer):
    def __init__(self, mask_flag=True, factor=10, scale=None, attention_dropout=0.1):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.dropout = tf.keras.layers.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top):
        # Q [B, H, L, D]
        _, H, L, E = K.shape
        _, _, S, _ = Q.shape
        B = tf.shape(Q)[0]

        # calculate the sampled Q_K， 增加一个维度后再复制 [b,h,l,l,emb]
        K_expand = tf.tile(tf.expand_dims(K, -3), (1, 1, L, 1, 1))

        # 对k的维度随机采样
        indx_q_seq = tf.random_uniform((S.value,), maxval=L, dtype=tf.int32)
        indx_k_seq = tf.random_uniform((sample_k,), maxval=L, dtype=tf.int32)
        K_sample = tf.gather(K_expand, tf.range(S), axis=2)
        K_sample = tf.gather(K_sample, indx_q_seq, axis=2)
        K_sample = tf.gather(K_sample, indx_k_seq, axis=3)  # 采样sample_k个 [b,h,l,sample,emb]

        # 矩阵乘法， [b,h,l,sample,emb]
        Q_K_sample = tf.squeeze(tf.matmul(tf.expand_dims(Q, -2), tf.transpose(K_sample, [0, 1, 2, 4, 3])), axis=-2)

        # 真正的Q采样，find the Top_k query with sparisty measurement [b,h,l] 公式部分，最大值-均值
        M = tf.reduce_max(Q_K_sample, axis=-1) - tf.divide(x=tf.reduce_sum(Q_K_sample, axis=-1),
                                                           y=tf.to_float(tf.convert_to_tensor(L)))
        M_top = tf.nn.top_k(M, n_top, sorted=False)[1]  # [b,h,sample]
        N_top = tf.nn.top_k(M, L, sorted=False)[1]  # [b,h,l]
        last_bott = N_top[:, :, n_top:]
        batch_indexes = tf.tile(tf.range(B)[:, tf.newaxis, tf.newaxis], (1, Q.shape[1], n_top))
        head_indexes = tf.tile(tf.range(Q.shape[1])[tf.newaxis, :, tf.newaxis], (B, 1, n_top))
        idx = tf.stack(values=[batch_indexes, head_indexes, M_top], axis=-1)  # todo [b,h,sample,3]

        # use the reduced Q to calculate Q_K  根据索引取值[b,h,sample,emb]
        Q_reduce = tf.gather_nd(Q, idx)

        Q_K = tf.matmul(Q_reduce, tf.transpose(K, [0, 1, 3, 2]))  # [b,h,sample,l]

        return Q_K, M_top, last_bott

    def _get_initial_context(self, V, L_Q):
        _, H, L_V, D = V.shape
        B = tf.shape(V)[0]

        if not self.mask_flag:
            V_sum = tf.reduce_sum(V, -2)  # [b,h,emb]
            contex = tf.identity(tf.tile(tf.expand_dims(V_sum, -2), [1, 1, L_Q, 1]))  # [b,h,l,emb]
        else:  # use mask
            assert (L_Q == L_V)  # requires that L_Q == L_V, i.e. for self-attention only
            contex = tf.math.cumsum(V, axis=-1)
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q, last_bott):
        _, H, L_V, D = V.shape
        B = tf.shape(V)[0]

        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores)

            # scores.masked_fill_(attn_mask.mask, -np.inf)
            num = 3.4 * math.pow(10, 38)
            scores = (scores * attn_mask.mask) + (-((attn_mask.mask * num + num) - num))

        attn = tf.keras.activations.softmax(scores, axis=-1)  # [b,h,l,sample]，将不平凡值softmax

        # todo 取索引  [b,h,sample]
        # batch_indexes = tf.tile(tf.range(B)[:, tf.newaxis, tf.newaxis], (1, V.shape[1], index.shape[-1]))
        # head_indexes = tf.tile(tf.range(V.shape[1])[tf.newaxis, :, tf.newaxis], (B, 1, index.shape[-1]))
        # idx = tf.stack(values=[batch_indexes, head_indexes, index], axis=-1)  # [b,h,sample,3]
        # part1 = tf.gather_nd(context_in, idx)

        batch_indexes2 = tf.tile(tf.range(B)[:, tf.newaxis, tf.newaxis], (1, V.shape[1], last_bott.shape[-1]))
        head_indexes2 = tf.tile(tf.range(V.shape[1])[tf.newaxis, :, tf.newaxis], (B, 1, last_bott.shape[-1]))
        idx2 = tf.stack(values=[batch_indexes2, head_indexes2, last_bott], axis=-1)  # [b,h,l-sample,3]

        # 对不平凡值做更新，其他的不变 [b,h,l,emb]
        part2 = tf.gather_nd(context_in, idx2)
        part1 = tf.matmul(attn, V)
        result = tf.concat([part1, part2], axis=2)
        return tf.convert_to_tensor(result)

    def call(self, inputs, attn_mask=None):
        queries, keys, values = inputs
        _, L, H, D = queries.shape
        _, S, _, _ = keys.shape
        B = tf.shape(queries)[0]

        queries = tf.reshape(queries, (B, H, L, D))
        keys = tf.reshape(keys, (B, H, S, D))
        values = tf.reshape(values, (B, H, S, D))

        U = self.factor
        u = self.factor
        print("u::", U)
        logging.info("u:{}".format(U))

        # 获取q和k的最相关的那部分，index是最相关部分的索引
        scores_top, index, last_bott = self._prob_QK(queries, keys, u, U)
        # add scale factor，scores_top是得到的filter的Q，index是所在的索引
        scale = self.scale or 1. / sqrt(D.value)
        if scale is not None:
            scores_top = scores_top * scale

        # get the context，处理values，将它的均值作为Q中平凡的值
        context = self._get_initial_context(values, L)
        # update the context with selected top_k queries，以前不平凡的值更新，平凡的值填充到Q
        context = self._update_context(context, values, scores_top, index, L, last_bott)

        return context  # [b,h,l,emb]


# Informer对多头自注意力机制的改进
class AttentionLayer(Layer):
    def __init__(self, attention, d_model, att_embedding_size, n_heads=4, d_keys=None,
                 d_values=None, use_positional_encoding=False):
        super(AttentionLayer, self).__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.att_embedding_size = att_embedding_size
        self.inner_attention = attention
        self.n_heads = n_heads

        self.use_positional_encoding = use_positional_encoding
        self.use_res = True
        self.use_layer_norm = True
        self.use_feed_forward = True
        self.dropout_rate = 0.2
        self.seed = 1001

    def build(self, input_shape):
        print(input_shape)
        B, L, _ = input_shape[0]
        _, S, _ = input_shape[1]
        embedding_size = int(input_shape[0][-1])
        self.W_Query = self.add_weight(name='query1', shape=[embedding_size, self.att_embedding_size * self.n_heads],
                                       dtype=tf.float32,
                                       initializer=TruncatedNormal(seed=self.seed))
        self.W_key = self.add_weight(name='key1', shape=[embedding_size, self.att_embedding_size * self.n_heads],
                                     dtype=tf.float32,
                                     initializer=TruncatedNormal(seed=self.seed + 1))
        self.W_Value = self.add_weight(name='value1', shape=[embedding_size, self.att_embedding_size * self.n_heads],
                                       dtype=tf.float32,
                                       initializer=TruncatedNormal(seed=self.seed + 2))
        if self.use_feed_forward:
            self.fw1 = self.add_weight('fw1', shape=[embedding_size, 4 * embedding_size], dtype=tf.float32,
                                       initializer=glorot_uniform(seed=self.seed))
            self.fw2 = self.add_weight('fw2', shape=[4 * embedding_size, embedding_size], dtype=tf.float32,
                                       initializer=glorot_uniform(seed=self.seed))
        self.dropout = Dropout(self.dropout_rate, seed=self.seed)
        self.ln = LayerNormalization()
        if self.use_positional_encoding:
            self.query_pe = PositionEncoding()
            self.key_pe = PositionEncoding()
            self.value_pe = PositionEncoding()

    def call(self, inputs, attn_mask=None):
        queries, keys, values = inputs
        _, L, D = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads
        B = tf.shape(queries)[0]

        if self.use_positional_encoding:
            queries = self.query_pe(queries)
            keys = self.key_pe(keys)
            values = self.value_pe(values)

        querys = tf.tensordot(queries, self.W_Query, axes=(-1, 0))
        keys = tf.tensordot(keys, self.W_key, axes=(-1, 0))
        values = tf.tensordot(values, self.W_Value, axes=(-1, 0))

        self.queries = tf.reshape(tf.split(querys, self.n_heads, axis=-1), [B, L, H, self.d_model // H])
        self.keys = tf.reshape(tf.split(keys, self.n_heads, axis=-1), [B, L, H, self.d_model // H])
        self.values = tf.reshape(tf.split(values, self.n_heads, axis=-1), [B, L, H, self.d_model // H])

        # [b,h,l,emb]->[b,l,h*emb]
        out = self.inner_attention([self.queries, self.keys, self.values], attn_mask=attn_mask)
        result = tf.reshape(out, (B, L, self.d_model))

        if self.use_res:  # 残差
            result += queries
        if self.use_layer_norm:  # 归一化
            result = self.ln(result)

        # [None,SL,EI]
        if self.use_feed_forward:  # 前馈
            fw1 = tf.nn.relu(tf.tensordot(result, self.fw1, axes=[-1, 0]))
            fw1 = self.dropout(fw1, training=True)
            fw2 = tf.tensordot(fw1, self.fw2, axes=[-1, 0])
            if self.use_res:
                result += fw2
            if self.use_layer_norm:
                result = self.ln(result)

        return result


# Pre-LN Transformer
class Pre_Transformer(Layer):

    def __init__(self, att_embedding_size=1, head_num=8, dropout_rate=0.0, use_positional_encoding=True, use_res=True,
                 use_feed_forward=True, use_layer_norm=True, blinding=True, seed=1024, supports_masking=False,
                 attention_type="scaled_dot_product", output_type="mean", **kwargs):
        super(Pre_Transformer, self).__init__(**kwargs)

        if head_num <= 0:
            raise ValueError('head_num must be a int > 0')
        self.att_embedding_size = att_embedding_size
        self.head_num = head_num
        self.num_units = att_embedding_size * head_num
        self.use_res = use_res
        self.use_feed_forward = use_feed_forward
        self.seed = seed
        self.use_positional_encoding = use_positional_encoding
        self.dropout_rate = dropout_rate
        self.use_layer_norm = use_layer_norm
        self.blinding = blinding
        self.attention_type = attention_type
        self.output_type = output_type
        self.supports_masking = supports_masking

    def build(self, input_shape):
        embedding_size = int(input_shape[0][-1])
        if self.num_units != embedding_size:
            raise ValueError(
                "att_embedding_size * head_num must equal the last dimension size of inputs,got %d * %d != %d" % (
                    self.att_embedding_size, self.head_num, embedding_size))
        self.seq_len_max = int(input_shape[0][-2])
        self.W_Query = self.add_weight(name='query', shape=[embedding_size, self.att_embedding_size * self.head_num],
                                       dtype=tf.float32,
                                       initializer=TruncatedNormal(seed=self.seed))
        self.W_key = self.add_weight(name='key', shape=[embedding_size, self.att_embedding_size * self.head_num],
                                     dtype=tf.float32,
                                     initializer=TruncatedNormal(seed=self.seed + 1))
        self.W_Value = self.add_weight(name='value', shape=[embedding_size, self.att_embedding_size * self.head_num],
                                       dtype=tf.float32,
                                       initializer=TruncatedNormal(seed=self.seed + 2))
        if self.attention_type == "additive":
            self.b = self.add_weight('b', shape=[self.att_embedding_size], dtype=tf.float32,
                                     initializer=glorot_uniform(seed=self.seed))
            self.v = self.add_weight('v', shape=[self.att_embedding_size], dtype=tf.float32,
                                     initializer=glorot_uniform(seed=self.seed))
        if self.use_feed_forward:
            self.fw1 = self.add_weight('fw1', shape=[self.num_units, 4 * self.num_units], dtype=tf.float32,
                                       initializer=glorot_uniform(seed=self.seed))
            self.fw2 = self.add_weight('fw2', shape=[4 * self.num_units, self.num_units], dtype=tf.float32,
                                       initializer=glorot_uniform(seed=self.seed))

        self.dropout = Dropout(self.dropout_rate, seed=self.seed)
        self.ln = LayerNormalization()

        # todo?
        K.set_learning_phase(True)

        if self.use_positional_encoding:
            self.query_pe = PositionEncoding()
            self.key_pe = PositionEncoding()

        # Be sure to call this somewhere!
        super(Pre_Transformer, self).build(input_shape)

    def call(self, inputs, mask=None, training=None, **kwargs):
        if self.supports_masking:
            queries, keys = inputs
            query_masks, key_masks = mask
            query_masks = tf.cast(query_masks, tf.float32)
            key_masks = tf.cast(key_masks, tf.float32)
        else:
            queries, keys, query_masks, key_masks = inputs  # [None,SL,EI], [None,SL]

        if self.use_positional_encoding:
            queries = self.query_pe(queries)
            keys = self.key_pe(queries)

        queries = self.ln(queries)
        keys = self.ln(keys)

        querys = tf.tensordot(queries, self.W_Query, axes=(-1, 0))
        keys = tf.tensordot(keys, self.W_key, axes=(-1, 0))
        values = tf.tensordot(keys, self.W_Value, axes=(-1, 0))  # [None,SL,EI]

        # [h*N,SL,EI/h]
        querys = tf.concat(tf.split(querys, self.head_num, axis=2), axis=0)
        keys = tf.concat(tf.split(keys, self.head_num, axis=2), axis=0)
        values = tf.concat(tf.split(values, self.head_num, axis=2), axis=0)

        if self.attention_type == "scaled_dot_product":  # 点积型注意力机制
            # [h*N,SL,EI/h]  SL序列长度
            outputs = tf.matmul(querys, keys, transpose_b=True)
            outputs = outputs / (keys.get_shape().as_list()[-1] ** 0.5)

        elif self.attention_type == "additive":  # 加型注意力机制
            querys_reshaped = tf.expand_dims(querys, axis=-2)
            keys_reshaped = tf.expand_dims(keys, axis=-3)
            outputs = tf.tanh(tf.nn.bias_add(querys_reshaped + keys_reshaped, self.b))
            outputs = tf.squeeze(tf.tensordot(outputs, tf.expand_dims(self.v, axis=-1), axes=[-1, 0]), axis=-1)
        else:
            raise ValueError("attention_type must be scaled_dot_product or additive")

        key_masks = tf.tile(key_masks, [self.head_num, 1])  # [h*N,SL]

        # (h*N, SL, SL)
        key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1])

        paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)

        # (h*N, T_q, T_k)
        outputs = tf.where(tf.equal(key_masks, 1), outputs, paddings)

        if self.blinding:
            try:
                outputs = tf.matrix_set_diag(outputs, tf.ones_like(outputs)[:, :, 0] * (-2 ** 32 + 1))
            except AttributeError:
                outputs = tf.compat.v1.matrix_set_diag(outputs, tf.ones_like(outputs)[:, :, 0] * (-2 ** 32 + 1))

        outputs -= reduce_max(outputs, axis=-1, keep_dims=True)
        outputs = softmax(outputs)
        query_masks = tf.tile(query_masks, [self.head_num, 1])  # (h*N, SL)
        # (h*N, SL, SL)
        query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]])

        outputs *= query_masks

        outputs = self.dropout(outputs, training=training)

        # Weighted sum
        # ( h*N, SL, EI/h)
        result = tf.matmul(outputs, values)
        result = tf.concat(tf.split(result, self.head_num, axis=0), axis=2)  # [None,SL,EI]

        if self.use_res:  # 残差
            result += queries

        if self.use_layer_norm:  # 归一化
            result = self.ln(result)
        # [None,SL,EI]
        if self.use_feed_forward:  # 前馈
            fw1 = tf.nn.relu(tf.tensordot(result, self.fw1, axes=[-1, 0]))
            fw1 = self.dropout(fw1, training=training)
            fw2 = tf.tensordot(fw1, self.fw2, axes=[-1, 0])
            if self.use_res:
                result += fw2

        if self.output_type == "mean":
            return reduce_mean(result, axis=1, keep_dims=True)  # [N,1,EI]
        elif self.output_type == "sum":
            return reduce_sum(result, axis=1, keep_dims=True)  # [N,1,EI]
        else:
            return result  # [None,SL,EI]


# Post-LN Transformer
class Transformer(Layer):

    def __init__(self, att_embedding_size=1, head_num=8, dropout_rate=0.0, use_positional_encoding=True, use_res=True,
                 use_feed_forward=True, use_layer_norm=True, blinding=True, seed=1024, supports_masking=False,
                 attention_type="scaled_dot_product", output_type="mean", **kwargs):
        super(Transformer, self).__init__(**kwargs)

        if head_num <= 0:
            raise ValueError('head_num must be a int > 0')
        self.att_embedding_size = att_embedding_size
        self.head_num = head_num
        self.num_units = att_embedding_size * head_num
        self.use_res = use_res
        self.use_feed_forward = use_feed_forward
        self.seed = seed
        self.use_positional_encoding = use_positional_encoding
        self.dropout_rate = dropout_rate
        self.use_layer_norm = use_layer_norm
        self.blinding = blinding
        self.attention_type = attention_type
        self.output_type = output_type
        self.supports_masking = supports_masking

    def build(self, input_shape):
        embedding_size = int(input_shape[0][-1])
        if self.num_units != embedding_size:
            raise ValueError(
                "att_embedding_size * head_num must equal the last dimension size of inputs,got %d * %d != %d" % (
                    self.att_embedding_size, self.head_num, embedding_size))
        self.seq_len_max = int(input_shape[0][-2])
        self.W_Query = self.add_weight(name='query', shape=[embedding_size, self.att_embedding_size * self.head_num],
                                       dtype=tf.float32,
                                       initializer=TruncatedNormal(seed=self.seed))
        self.W_key = self.add_weight(name='key', shape=[embedding_size, self.att_embedding_size * self.head_num],
                                     dtype=tf.float32,
                                     initializer=TruncatedNormal(seed=self.seed + 1))
        self.W_Value = self.add_weight(name='value', shape=[embedding_size, self.att_embedding_size * self.head_num],
                                       dtype=tf.float32,
                                       initializer=TruncatedNormal(seed=self.seed + 2))
        if self.attention_type == "additive":
            self.b = self.add_weight('b', shape=[self.att_embedding_size], dtype=tf.float32,
                                     initializer=glorot_uniform(seed=self.seed))
            self.v = self.add_weight('v', shape=[self.att_embedding_size], dtype=tf.float32,
                                     initializer=glorot_uniform(seed=self.seed))
        if self.use_feed_forward:
            self.fw1 = self.add_weight('fw1', shape=[self.num_units, 4 * self.num_units], dtype=tf.float32,
                                       initializer=glorot_uniform(seed=self.seed))
            self.fw2 = self.add_weight('fw2', shape=[4 * self.num_units, self.num_units], dtype=tf.float32,
                                       initializer=glorot_uniform(seed=self.seed))

        self.dropout = Dropout(self.dropout_rate, seed=self.seed)
        self.ln = LayerNormalization()

        # todo?
        K.set_learning_phase(True)

        if self.use_positional_encoding:
            self.query_pe = PositionEncoding()
            self.key_pe = PositionEncoding()

        # Be sure to call this somewhere!
        super(Transformer, self).build(input_shape)

    def call(self, inputs, mask=None, training=None, **kwargs):
        if self.supports_masking:
            queries, keys = inputs
            query_masks, key_masks = mask
            query_masks = tf.cast(query_masks, tf.float32)
            key_masks = tf.cast(key_masks, tf.float32)
        else:
            queries, keys, query_masks, key_masks = inputs  # [None,SL,EI], [None,SL]

        if self.use_positional_encoding:
            queries = self.query_pe(queries)
            keys = self.key_pe(queries)

        querys = tf.tensordot(queries, self.W_Query, axes=(-1, 0))
        keys = tf.tensordot(keys, self.W_key, axes=(-1, 0))
        values = tf.tensordot(keys, self.W_Value, axes=(-1, 0))  # [None,SL,EI]

        # [h*N,SL,EI/h]
        querys = tf.concat(tf.split(querys, self.head_num, axis=2), axis=0)
        keys = tf.concat(tf.split(keys, self.head_num, axis=2), axis=0)
        values = tf.concat(tf.split(values, self.head_num, axis=2), axis=0)

        if self.attention_type == "scaled_dot_product":  # 点积型注意力机制
            # [h*N,SL,EI/h]  SL序列长度
            outputs = tf.matmul(querys, keys, transpose_b=True)
            outputs = outputs / (keys.get_shape().as_list()[-1] ** 0.5)

        elif self.attention_type == "additive":  # 加型注意力机制
            querys_reshaped = tf.expand_dims(querys, axis=-2)
            keys_reshaped = tf.expand_dims(keys, axis=-3)
            outputs = tf.tanh(tf.nn.bias_add(querys_reshaped + keys_reshaped, self.b))
            outputs = tf.squeeze(tf.tensordot(outputs, tf.expand_dims(self.v, axis=-1), axes=[-1, 0]), axis=-1)
        else:
            raise ValueError("attention_type must be scaled_dot_product or additive")

        key_masks = tf.tile(key_masks, [self.head_num, 1])  # [h*N,SL]

        # (h*N, SL, SL)
        key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1])

        paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)

        # (h*N, T_q, T_k)
        outputs = tf.where(tf.equal(key_masks, 1), outputs, paddings)

        if self.blinding:
            try:
                outputs = tf.matrix_set_diag(outputs, tf.ones_like(outputs)[:, :, 0] * (-2 ** 32 + 1))
            except AttributeError:
                outputs = tf.compat.v1.matrix_set_diag(outputs, tf.ones_like(outputs)[:, :, 0] * (-2 ** 32 + 1))

        outputs -= reduce_max(outputs, axis=-1, keep_dims=True)
        outputs = softmax(outputs)
        query_masks = tf.tile(query_masks, [self.head_num, 1])  # (h*N, SL)
        # (h*N, SL, SL)
        query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]])

        outputs *= query_masks

        outputs = self.dropout(outputs, training=training)

        # Weighted sum
        # ( h*N, SL, EI/h)
        result = tf.matmul(outputs, values)
        result = tf.concat(tf.split(result, self.head_num, axis=0), axis=2)  # [None,SL,EI]

        if self.use_res:  # 残差
            result += queries
        if self.use_layer_norm:  # 归一化
            result = self.ln(result)

        # [None,SL,EI]
        if self.use_feed_forward:  # 前馈
            fw1 = tf.nn.relu(tf.tensordot(result, self.fw1, axes=[-1, 0]))
            fw1 = self.dropout(fw1, training=training)
            fw2 = tf.tensordot(fw1, self.fw2, axes=[-1, 0])
            if self.use_res:
                result += fw2
            if self.use_layer_norm:
                result = self.ln(result)

        if self.output_type == "mean":
            return reduce_mean(result, axis=1, keep_dims=True)  # [N,1,EI]
        elif self.output_type == "sum":
            return reduce_sum(result, axis=1, keep_dims=True)  # [N,1,EI]
        else:
            return result  # [None,SL,EI]


class LayerNormalization(Layer):
    def __init__(self, axis=-1, eps=1e-9, center=True,
                 scale=True, **kwargs):
        self.axis = axis
        self.eps = eps
        self.center = center
        self.scale = scale
        super(LayerNormalization, self).__init__(**kwargs)

    def build(self, input_shape):
        self.gamma = self.add_weight(name='gamma', shape=input_shape[-1:],
                                     initializer=Ones(), trainable=True)
        self.beta = self.add_weight(name='beta', shape=input_shape[-1:],
                                    initializer=Zeros(), trainable=True)
        super(LayerNormalization, self).build(input_shape)

    def call(self, inputs):
        mean = K.mean(inputs, axis=self.axis, keepdims=True)
        variance = K.mean(K.square(inputs - mean), axis=-1, keepdims=True)
        std = K.sqrt(variance + self.eps)
        outputs = (inputs - mean) / std
        if self.scale:
            outputs *= self.gamma
        if self.center:
            outputs += self.beta
        return outputs


class PositionEncoding(Layer):
    def __init__(self, pos_embedding_trainable=True,
                 zero_pad=False, scale=True, **kwargs):
        self.pos_embedding_trainable = pos_embedding_trainable
        self.zero_pad = zero_pad
        self.scale = scale
        super(PositionEncoding, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        _, T, num_units = input_shape.as_list()  # inputs.get_shape().as_list()
        position_enc = np.array([
            [pos / np.power(10000, 2. * (i // 2) / num_units) for i in range(num_units)]
            for pos in range(T)])

        # Second part, apply the cosine to even columns and sin to odds.
        position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # dim 2i
        position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # dim 2i+1
        if self.zero_pad:
            position_enc[0, :] = np.zeros(num_units)
        self.lookup_table = self.add_weight("lookup_table", (T, num_units),
                                            initializer=identity(position_enc),
                                            trainable=self.pos_embedding_trainable)

        # Be sure to call this somewhere!
        super(PositionEncoding, self).build(input_shape)

    def call(self, inputs, mask=None):
        _, T, num_units = inputs.get_shape().as_list()
        position_ind = tf.expand_dims(tf.range(T), 0)
        outputs = tf.nn.embedding_lookup(self.lookup_table, position_ind)
        if self.scale:
            outputs = outputs * num_units ** 0.5
        return outputs + inputs


# BiasEncoding
class BiasEncoding(Layer):
    def __init__(self, sess_max_count, seed=1024, **kwargs):
        self.sess_max_count = sess_max_count
        self.seed = seed
        super(BiasEncoding, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.

        if self.sess_max_count == 1:
            embed_size = input_shape[2].value
            seq_len_max = input_shape[1].value
        else:
            try:
                embed_size = input_shape[0][2].value
                seq_len_max = input_shape[0][1].value
            except AttributeError:
                embed_size = input_shape[0][2]
                seq_len_max = input_shape[0][1]

        self.sess_bias_embedding = self.add_weight('sess_bias_embedding', shape=(self.sess_max_count, 1, 1),
                                                   initializer=TruncatedNormal(
                                                       mean=0.0, stddev=0.0001, seed=self.seed))
        self.seq_bias_embedding = self.add_weight('seq_bias_embedding', shape=(1, seq_len_max, 1),
                                                  initializer=TruncatedNormal(
                                                      mean=0.0, stddev=0.0001, seed=self.seed))
        self.item_bias_embedding = self.add_weight('item_bias_embedding', shape=(1, 1, embed_size),
                                                   initializer=TruncatedNormal(
                                                       mean=0.0, stddev=0.0001, seed=self.seed))

        # Be sure to call this somewhere!
        super(BiasEncoding, self).build(input_shape)

    def call(self, inputs, mask=None):
        """
        :param concated_embeds_value: None * field_size * embedding_size
        :return: None*1
        """
        transformer_out = []
        for i in range(self.sess_max_count):
            transformer_out.append(
                inputs[i] + self.item_bias_embedding + self.seq_bias_embedding + self.sess_bias_embedding[i])
        return transformer_out


"""
（1）
tf.gather_nd：gather_nd 实现了根据指定的 参数 indices 来提取params 的元素重建出一个tensor。
（2）
tf.scatter_nd_update：https://blog.csdn.net/DaVinciL/article/details/84027241
"""
