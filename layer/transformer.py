import numpy as np
import tensorflow as tf
from tensorflow.python.keras.layers import Layer, Dropout, Dense
from tensorflow.python.keras import backend as K
from tensorflow.python.ops.init_ops import Zeros, Ones
from tensorflow.python.ops.init_ops import TruncatedNormal, glorot_uniform_initializer as glorot_uniform, \
    identity_initializer as identity


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


class LayerNormalization(Layer):
    def __init__(self, axis=-1, eps=1e-9, center=True, scale=True, **kwargs):
        super(LayerNormalization, self).__init__(**kwargs)
        self.axis = axis
        self.eps = eps
        self.center = center
        self.scale = scale

    def build(self, input_shape):
        self.gamma = self.add_weight(name='gamma', shape=input_shape[-1:],
                                     initializer=Ones(), trainable=True)
        self.beta = self.add_weight(name='beta', shape=input_shape[-1:],
                                    initializer=Zeros(), trainable=True)
        super(LayerNormalization, self).build(input_shape)

    def call(self, inputs, **kwargs):
        mean = K.mean(inputs, axis=self.axis, keepdims=True)
        variance = K.mean(K.square(inputs - mean), axis=-1, keepdims=True)
        std = K.sqrt(variance + self.eps)
        outputs = (inputs - mean) / std
        if self.scale:
            outputs *= self.gamma
        if self.center:
            outputs += self.beta
        return outputs


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


class MultiHeadAttention(Layer):
    def __init__(self, d_model, num_heads, attention_dropout_rate):
        super().__init__()

        assert d_model % num_heads == 0

        # Define query, key, value matrix
        self._wq = Dense(d_model, use_bias=False)  # matrix shape: (d_model, d_model)
        self._wk = Dense(d_model, use_bias=False)  # matrix shape: (d_model, d_model)
        self._wv = Dense(d_model, use_bias=False)  # matrix shape: (d_model, d_model)

        # Output dense layer
        self._dense = Dense(d_model, use_bias=False)  # matrix shape: (d_model, d_model)

        # Dropout
        self._attention_dropout = Dropout(attention_dropout_rate)

        # Define other attributes
        self._d_model = d_model
        self._num_heads = num_heads
        self._d_k = d_model // num_heads

    def call(self, inputs, **kwargs):
        q, k, v, mask = inputs
        """
        Args:
            q: shape == (batch_size, seq_len_tgt, d_model)
            k: shape == (batch_size, seq_len_src, d_model)
            v: shape == (batch_size, seq_len_src, d_model)

        Output: shape == (batch_size, seq_len_tgt, d_model)
        """
        batch_size = tf.shape(q)[0]
        seq_len = q.get_shape().as_list()[1]

        # Convert inputs to query, key, value
        q = self._wq(q)  # output shape: (batch_size, seq_len, d_model)
        k = self._wk(k)  # output shape: (batch_size, seq_len, d_model)
        v = self._wv(v)  # output shape: (batch_size, seq_len, d_model)

        # Split head into num_heads
        q = self._split_heads(batch_size, q)  # output shape: (batch_size, num_heads, seq_len, d_k)
        k = self._split_heads(batch_size, k)  # output shape: (batch_size, num_heads, seq_len, d_k)
        v = self._split_heads(batch_size, v)  # output shape: (batch_size, num_heads, seq_len, d_k)

        # batch_size, num_heads, seq_len, d_k)
        attn = scaled_dot_product_attention(q, k, v, mask)
        attn = tf.transpose(attn, perm=[0, 2, 1, 3])  # output shape: (batch_size, seq_len, num_heads, d_k)

        # Concat attention heads (merge 2nd and 3rd dimensions into one), output shape: (batch_size, seq_len, d_model)
        concat_attn = tf.reshape(attn, (batch_size, seq_len, self._d_model))

        # Apply final dense layer
        output = self._dense(concat_attn)  # output shape: (batch_size, seq_len, d_model)

        return output

    def _split_heads(self, batch_size, tensor):
        """
        Args:
            tensor: shape == (batch_size, seq_len, d_model)

        Returns: shape == (batch_size, num_heads, seq_len, d_k)

        """
        # Reshape the last dimention: (d_model,) -> (num_heads, d_k)
        seq_len = tensor.get_shape().as_list()[1]
        x = tf.reshape(tensor, (batch_size, seq_len, self._num_heads, self._d_k))
        # Replace the dimension
        tx = tf.transpose(x, perm=[0, 2, 1, 3])  # (batch_size, num_heads, seq_len, d_k)
        return tx


class PointwiseFeedForwardNetwork(Layer):
    def __init__(self, d_model, d_ff, activation):
        super().__init__()
        self._mid = Dense(units=d_ff, activation=activation)
        self._last = Dense(units=d_model)

    def call(self, inputs, **kwargs):
        mid = self._mid(inputs)
        output = self._last(mid)
        return output


# ====== Define PostLN model ======
class PostLN(Layer):
    def __init__(self, d_model, num_heads, residual_dropout_rate=0.2,
                 attention_dropout_rate=0.2, activation='relu'):
        super().__init__()

        self._mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads,
                                       attention_dropout_rate=attention_dropout_rate)
        self._fst_layernorm = LayerNormalization()
        self._fst_dropout = Dropout(residual_dropout_rate)
        d_ff = 4 * d_model
        self._ffn = PointwiseFeedForwardNetwork(d_model=d_model, d_ff=d_ff, activation=activation)
        self._snd_layernorm = LayerNormalization()
        self._snd_dropout = Dropout(rate=residual_dropout_rate)
        self.q_pos_encod = PositionEncoding()
        self.k_pos_encod = PositionEncoding()

    def call(self, inputs, training=None, look_ahead_mask=None, **kwargs):
        queries, keys, query_masks, key_masks = inputs

        # 【1】位置编码
        queries = self.q_pos_encod(queries)
        keys = self.k_pos_encod(keys)

        # 【2】多头自注意力
        attn = self._mha([queries, keys, keys, query_masks])
        # dropout，残差，归一化
        attn = self._fst_dropout(attn, training=training)
        fst_out = self._fst_layernorm(queries + attn)

        # 【3】前馈，dropout，残差，归一化（Residual dropout: LayerNorm(x + Dropout(Sublayer(x)))）
        ffn = self._ffn(fst_out)
        ffn = self._snd_dropout(ffn, training=training)
        snd_out = self._snd_layernorm(fst_out + ffn)

        return snd_out


# ====== Define PreLN model ======

class PreLN(Layer):
    def __init__(self, d_model, num_heads, residual_dropout_rate=0.2,
                 attention_dropout_rate=0.2, activation='relu'):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.residual_dropout_rate = residual_dropout_rate
        self.attention_dropout_rate = attention_dropout_rate
        self.activation = activation

    def build(self, input_shape):
        K.set_learning_phase(True)

        self._mha = MultiHeadAttention(d_model=self.d_model, num_heads=self.num_heads,
                                       attention_dropout_rate=self.attention_dropout_rate)
        self.q_fst_layernorm = LayerNormalization()
        self.k_fst_layernorm = LayerNormalization()
        self._fst_dropout = Dropout(self.residual_dropout_rate)
        d_ff = 4 * self.d_model
        self._ffn = PointwiseFeedForwardNetwork(d_model=self.d_model, d_ff=d_ff, activation=self.activation)
        self._snd_layernorm = LayerNormalization()
        self._snd_dropout = Dropout(self.residual_dropout_rate)
        self.q_pos_encod = PositionEncoding()
        self.k_pos_encod = PositionEncoding()

    def call(self, inputs, training=None, look_ahead_mask=None, **kwargs):
        queries, keys, query_masks, key_masks = inputs

        # 【1】位置编码
        queries = self.q_pos_encod(queries)
        keys = self.k_pos_encod(keys)

        # 【2】归一化，多头自注意力
        queries = self.q_fst_layernorm(queries)
        keys = self.k_fst_layernorm(keys)
        attn = self._mha([queries, keys, keys, query_masks])
        # dropout，残差，Residual dropout: LayerNorm(x + Dropout(Sublayer(x)))
        attn = self._fst_dropout(attn, training=training)
        fst_out = queries + attn

        # 【4】归一化，前馈
        y = self._snd_layernorm(fst_out)
        ffn = self._ffn(y)
        # dropout，残差，Residual dropout: LayerNorm(x + Dropout(Sublayer(x)))
        ffn = self._snd_dropout(ffn, training=training)
        snd_out = fst_out + ffn

        return snd_out
