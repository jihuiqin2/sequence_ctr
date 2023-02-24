import tensorflow as tf

from layer.utils import layer_norm, prelu, reduce_sum


# 注意力（query，key）dien,bst,din,
def attention_v1(query, keys, mask=None, stag='null', mode='SUM', softmax_stag=1, return_alphas=False):
    # In case of Bi-RNN, concatenate the forward and the backward RNN outputs.
    if isinstance(keys, tuple):
        keys = tf.concat(keys, 2)
        query = tf.concat(values=[query, query, ], axis=1)

    if len(keys.get_shape().as_list()) == 2:
        keys = tf.expand_dims(keys, 1)

    facts_size = keys.get_shape().as_list()[-1]
    querry_size = query.get_shape().as_list()[-1]

    query = tf.layers.dense(query, facts_size, activation=None, name='f1' + stag)
    query = prelu(query)
    queries = tf.tile(query, [1, tf.shape(keys)[1]])  # [N,SL*EI]
    queries = tf.reshape(queries, tf.shape(keys))  # [N,SL,EI]
    din_all = tf.concat([queries, keys, queries - keys, queries * keys], axis=-1)

    # dnn layer
    d_layer_1_all = tf.layers.dense(din_all, 80, activation=tf.nn.sigmoid, name='f1_att' + stag)
    d_layer_2_all = tf.layers.dense(d_layer_1_all, 40, activation=tf.nn.sigmoid, name='f2_att' + stag)
    d_layer_3_all = tf.layers.dense(d_layer_2_all, 1, activation=None, name='f3_att' + stag)
    d_layer_3_all = tf.reshape(d_layer_3_all, [-1, 1, tf.shape(keys)[1]])
    scores = d_layer_3_all  # [N,1,SL]

    if mask is not None:
        mask = tf.equal(mask, tf.ones_like(mask))
        key_masks = tf.expand_dims(mask, 1)  # [B, 1, SL]
        paddings = tf.ones_like(scores) * (-2 ** 32 + 1)
        scores = tf.where(key_masks, scores, paddings)  # [B, 1, SL]

    # Scale
    # scores = scores / (keys.get_shape().as_list()[-1] ** 0.5)

    # Activation
    if softmax_stag:
        scores = tf.nn.softmax(scores)  # [B, 1, SL]

    # Weighted sum
    if mode == 'SUM':
        output = tf.matmul(scores, keys)  # [B, 1, EI]
    else:
        scores = tf.reshape(scores, [-1, tf.shape(keys)[1]])  # [N,SL]
        output = keys * tf.expand_dims(scores, -1)  # [N,SL,EI]
        output = tf.reshape(output, tf.shape(keys))
    if return_alphas:
        return output, scores  # [N,SL,EI]  [N,SL]
    return output


# 多头注意力（query, head_key, pos_emb）dmin,
def multi_attention(query, head_key, pos_emb, mask, stag='null', mode='SUM', softmax_stag=1,
                    time_major=False, return_alphas=False):
    # In case of Bi-RNN, concatenate the forward and the backward RNN outputs.
    if isinstance(head_key, tuple):
        head_key = tf.concat(head_key, 2)
        query = tf.concat(values=[query, query], axis=1)

    if time_major:
        head_key = tf.array_ops.transpose(head_key, [1, 0, 2])

    facts_size = head_key.get_shape().as_list()[-1]
    querry_size = query.get_shape().as_list()[-1]

    queries = tf.tile(query, [1, tf.shape(head_key)[1]])
    queries = tf.reshape(queries, tf.shape(head_key))  # [N,SL,EI]

    if pos_emb is None:
        queries = queries
    else:
        queries = tf.concat([queries, pos_emb], axis=-1)  # 位置编码和目标项拼接

    queries = tf.layers.dense(queries, head_key.get_shape().as_list()[-1], activation=None, name=stag + 'dmr1')
    din_all = tf.concat([queries, head_key, queries - head_key, queries * head_key], axis=-1)

    # dnn layer
    d_layer_1_all = tf.layers.dense(din_all, 80, activation=tf.nn.sigmoid, name='f1_att' + stag)
    d_layer_2_all = tf.layers.dense(d_layer_1_all, 40, activation=tf.nn.sigmoid, name='f2_att' + stag)
    d_layer_3_all = tf.layers.dense(d_layer_2_all, 1, activation=None, name='f3_att' + stag)
    d_layer_3_all = tf.reshape(d_layer_3_all, [-1, 1, tf.shape(head_key)[1]])
    scores = d_layer_3_all  # [B, 1, SL]

    # Mask
    # key_masks = tf.sequence_mask(facts_length, tf.shape(head_key)[1])   # [B, T]
    mask = tf.equal(mask, tf.ones_like(mask))
    key_masks = tf.expand_dims(mask, 1)  # [B, 1, SL]
    paddings = tf.ones_like(scores) * (-2 ** 32 + 1)
    scores = tf.where(key_masks, scores, paddings)  # [B, 1, SL]

    paddings_no_softmax = tf.zeros_like(scores)
    scores_no_softmax = tf.where(key_masks, scores, paddings_no_softmax)

    # Scale
    # scores = scores / (head_key.get_shape().as_list()[-1] ** 0.5)

    # Activation
    if softmax_stag:
        scores = tf.nn.softmax(scores)  # [B, 1, SL]

    # Weighted sum
    if mode == 'SUM':
        output = tf.matmul(scores, head_key)  # [B, 1, EI]
    else:
        scores = tf.reshape(scores, [-1, tf.shape(head_key)[1]])
        output = head_key * tf.expand_dims(scores, -1)
        output = tf.reshape(output, tf.shape(head_key))
    return output, scores, scores_no_softmax


# 多头自注意力1（inputs）DMIN
def self_multi_head_attn_v1(inputs, num_dim, num_heads, dropout_rate, name="", is_training=True, is_layer_norm=True):
    """
    Args:
      inputs(query): A 3d tensor with shape of [N, T_q, C_q]
      inputs(keys): A 3d tensor with shape of [N, T_k, C_k]
    """
    Q_K_V = tf.layers.dense(inputs, 3 * num_dim)  # tf.nn.relu
    Q, K, V = tf.split(Q_K_V, 3, -1)  # [None,SL,EI]

    Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)  # (h*N, SL, EI/h)
    K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)  # (h*N, SL, EI/h)
    V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)  # (h*N, SL, EI/h)
    print('V_.get_shape()', V_.get_shape().as_list())

    # (h*N, T_q, T_k)
    outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))  # [h*N, SL, SL]
    align = outputs / (36 ** 0.5)  # 归一化

    print('align.get_shape()', align.get_shape().as_list())
    diag_val = tf.ones_like(align[0, :, :])  # [SL, SL]
    tril = tf.contrib.linalg.LinearOperatorTriL(diag_val).to_dense()  # 下三角矩阵

    # 扩展维度[1,SL,SL],然后平铺[None,SL,SL]
    key_masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(align)[0], 1, 1])
    padding = tf.ones_like(key_masks) * (-2 ** 32 + 1)
    outputs = tf.where(tf.equal(key_masks, 0), padding, align)  # [h*N, SL, SL]

    outputs = tf.nn.softmax(outputs)
    outputs = tf.layers.dropout(outputs, dropout_rate, training=is_training)
    outputs = tf.matmul(outputs, V_)  # [h*N, SL, EI/h]
    outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)  # (N, SL, EI)
    # output linear
    outputs = tf.layers.dense(outputs, num_dim)

    # drop_out before residual and layer_normal
    outputs = tf.layers.dropout(outputs, dropout_rate, training=is_training)
    # Residual connection
    outputs += inputs  # (N, SL, EI)
    # Normalize
    if is_layer_norm:
        outputs = layer_norm(outputs, name=name)  # (N, SL, EI)

    return outputs


# 多头自注意力2（inputs）DMIN
def self_multi_head_attn_v2(inputs, num_dim, num_heads, dropout_rate, name="", is_training=True, is_layer_norm=True):
    """
    Args:
      inputs(query): A 3d tensor with shape of [N, T_q, C_q]
      inputs(keys): A 3d tensor with shape of [N, T_k, C_k]
    """
    Q_K_V = tf.layers.dense(inputs, 3 * num_dim)  # tf.nn.relu
    Q, K, V = tf.split(Q_K_V, 3, -1)

    Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)  # (h*N, SL, EI/h)
    K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)  # (h*N, SL, EI/h)
    V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)  # (h*N, SL, EI/h)
    print('V_.get_shape()', V_.get_shape().as_list())

    outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))  # [h*N, SL, SL]
    align = outputs / (36 ** 0.5)

    print('align.get_shape()', align.get_shape().as_list())
    diag_val = tf.ones_like(align[0, :, :])  # [SL, SL]
    tril = tf.contrib.linalg.LinearOperatorTriL(diag_val).to_dense()  # [SL, SL] 下三角矩阵

    # 扩展维度[1,SL,SL] 然后平铺[N,SL,SL]
    key_masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(align)[0], 1, 1])  # [N,SL,SL]
    padding = tf.ones_like(key_masks) * (-2 ** 32 + 1)
    outputs = tf.where(tf.equal(key_masks, 0), padding, align)  # [h*N, SL, SL]

    outputs = tf.nn.softmax(outputs)
    outputs = tf.layers.dropout(outputs, dropout_rate, training=is_training)
    outputs = tf.matmul(outputs, V_)  # (h*N, SL, EI/h)

    # Restore shape h个[N,SL,EI/h]
    outputs1 = tf.split(outputs, num_heads, axis=0)
    outputs2 = []
    for head_index, outputs3 in enumerate(outputs1):
        outputs3 = tf.layers.dense(outputs3, num_dim)  # 维度变化，[N,SL,EI]
        outputs3 = tf.layers.dropout(outputs3, dropout_rate, training=is_training)
        outputs3 += inputs
        print("outputs3.get_shape()", outputs3.get_shape())
        if is_layer_norm:
            outputs3 = layer_norm(outputs3, name=name + str(head_index))  # (N, SL, EI)
        outputs2.append(outputs3)

    return outputs2  # list h个[N,SL,EI]


def scaled_dot_product_attention(q, k, v, mask):
    # mask value should be smaller when fp16 is enabled https://github.com/NVIDIA/apex/issues/93
    if q.dtype == tf.float32:
        mask_val = -1e9
    elif q.dtype == tf.float16:
        mask_val = -1e4
    else:
        raise Exception(f"Input type {q.dtype} is not float16 or float32")

    positionwise_score = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

    # Scaling by the embed dim
    d_k = tf.cast(tf.shape(k)[-1], k.dtype)
    scaled_positionwise_score = positionwise_score / tf.sqrt(d_k)

    if mask is not None:
        mask = tf.cast(mask, scaled_positionwise_score.dtype)
        scaled_positionwise_score += (mask * mask_val)

    # Calculate weight  [b,h,sl,sl]
    weights = tf.nn.softmax(scaled_positionwise_score, dim=-1)

    # Attention dropout
    # weights = attention_dropout(weights)

    output = weights @ v  # dim: (b,h,seq_len_q, d_k)

    return output


"""
1】 
tf.contrib.linalg.LinearOperatorTriL(diag_val).to_dense() 下三角矩阵
[[1., 0.]
[1., 1.]]

2】
tf.shape()和tensor.get_shape()区别
https://blog.csdn.net/Laox1ao/article/details/79896656

3】
tf.expand_dims()函数用于给函数增加维度。
https://www.cnblogs.com/yibeimingyue/p/15128733.html

4】
tf.tile()平铺，用于在同一维度上的复制
https://blog.csdn.net/weixin_41089007/article/details/93924098

5】
tf.matmul矩阵乘法
"""
