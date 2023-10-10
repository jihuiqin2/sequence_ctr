import itertools

import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.backend import batch_dot

from tensorflow.python.ops.init_ops import Zeros, Ones, Constant, TruncatedNormal, \
    glorot_normal_initializer as glorot_normal, \
    glorot_uniform_initializer as glorot_uniform

from tensorflow.python.keras.layers import Layer, MaxPooling2D, Conv2D, Dropout, Lambda, Dense, Flatten
from tensorflow.python.keras.regularizers import l2

from layer.utils import reduce_mean
from layer.utils import reduce_sum


class SENETLayer(Layer):
    def __init__(self, reduction_ratio=3, seed=1024, **kwargs):
        self.reduction_ratio = reduction_ratio

        self.seed = seed
        super(SENETLayer, self).__init__(**kwargs)

    def build(self, input_shape):  # [b,sl,ei]
        if not isinstance(input_shape, list) or len(input_shape) < 2:
            raise ValueError('A `AttentionalFM` layer should be called '
                             'on a list of at least 2 inputs')

        self.filed_size = len(input_shape)
        self.embedding_size = input_shape[0][-1]
        reduction_size = max(1, self.filed_size // self.reduction_ratio)

        self.W_1 = self.add_weight(shape=(
            self.filed_size, reduction_size), initializer=glorot_normal(seed=self.seed), name="W_1")
        self.W_2 = self.add_weight(shape=(
            reduction_size, self.filed_size), initializer=glorot_normal(seed=self.seed), name="W_2")

        self.tensordot = Lambda(
            lambda x: tf.tensordot(x[0], x[1], axes=(-1, 0)))

        # Be sure to call this somewhere!
        super(SENETLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        if K.ndim(inputs[0]) != 3:
            raise ValueError(
                "Unexpected inputs dimensions %d, expect to be 3 dimensions" % (K.ndim(inputs)))
        inputs = tf.concat(inputs, axis=1)
        Z = reduce_mean(inputs, axis=-1)  # [b,sl]
        A_1 = tf.nn.relu(self.tensordot([Z, self.W_1]))
        A_2 = tf.nn.relu(self.tensordot([A_1, self.W_2]))
        V = tf.multiply(inputs, tf.expand_dims(A_2, axis=2))  # [b,sl,ei]
        split_V = tf.split(V, self.filed_size, axis=1)  # 转为list sl个[b,1,ei]

        return split_V


class SENETLayer2(Layer):
    def __init__(self, reduction_ratio=3, seed=1024, **kwargs):
        self.reduction_ratio = reduction_ratio

        self.seed = seed
        super(SENETLayer2, self).__init__(**kwargs)

    def build(self, input_shape):  # [b,sl,ei]

        self.filed_size = input_shape[1]
        self.embedding_size = input_shape[-1]
        reduction_size = 1

        self.W_1 = self.add_weight(shape=(
            self.filed_size, reduction_size), initializer=glorot_normal(seed=self.seed), name="W_1")
        self.W_2 = self.add_weight(shape=(
            reduction_size, self.filed_size), initializer=glorot_normal(seed=self.seed), name="W_2")

        self.tensordot = Lambda(
            lambda x: tf.tensordot(x[0], x[1], axes=(-1, 0)))

        # Be sure to call this somewhere!
        super(SENETLayer2, self).build(input_shape)

    def call(self, inputs, **kwargs):
        Z = reduce_mean(inputs, axis=-1)  # [b,sl]
        A_1 = tf.nn.relu(self.tensordot([Z, self.W_1]))
        A_2 = tf.nn.relu(self.tensordot([A_1, self.W_2]))
        V = tf.multiply(inputs, tf.expand_dims(A_2, axis=2))  # [b,sl,ei]

        return V


class SKNETLayer2(Layer):
    def __init__(self, reduction_ratio=3, seed=1024, **kwargs):
        self.reduction_ratio = reduction_ratio

        self.seed = seed
        super(SKNETLayer2, self).__init__(**kwargs)

    def build(self, input_shape):  # [b,sl,ei]

        self.filed_size = input_shape[1]
        self.embedding_size = input_shape[-1]
        reduction_size = 1

        self.W_1 = self.add_weight(shape=(
            self.filed_size, reduction_size), initializer=glorot_normal(seed=self.seed), name="W_1")
        self.W_2 = self.add_weight(shape=(
            reduction_size, self.filed_size), initializer=glorot_normal(seed=self.seed), name="W_2")

        self.W_4 = self.add_weight(shape=(
            self.embedding_size, self.embedding_size), initializer=glorot_normal(seed=self.seed), name="W_4")
        self.W_5 = self.add_weight(shape=(
            self.embedding_size, self.embedding_size), initializer=glorot_normal(seed=self.seed), name="W_5")

        self.tensordot = Lambda(
            lambda x: tf.tensordot(x[0], x[1], axes=(-1, 0)))

        self.projection_a = self.add_weight(shape=(self.embedding_size, 1),
                                            initializer=glorot_normal(seed=12006), name="projection_a")
        self.projection_b = self.add_weight(shape=(self.embedding_size, 1),
                                            initializer=glorot_normal(seed=12006), name="projection_b")

        # Be sure to call this somewhere!
        super(SKNETLayer2, self).build(input_shape)

    def call(self, inputs, **kwargs):
        inputs1 = tf.nn.relu(self.tensordot([inputs, self.W_4]))
        inputs2 = tf.nn.relu(self.tensordot([inputs, self.W_5]))
        inputs_a = inputs1 + inputs2
        Z = reduce_mean(inputs_a, axis=-1)  # [b,sl]
        A_1 = tf.nn.relu(self.tensordot([Z, self.W_1]))
        A_2 = tf.nn.relu(self.tensordot([A_1, self.W_2]))
        V = tf.multiply(inputs, tf.expand_dims(A_2, axis=2))  # [b,sl,ei]

        attr_1 = tf.nn.softmax(tf.tensordot(V, self.projection_a, axes=(-1, 0)), dim=1)  # [b,1]
        attr_2 = tf.nn.softmax(tf.tensordot(V, self.projection_b, axes=(-1, 0)), dim=1)  # [b,1]

        h1 = tf.multiply(inputs1, attr_1)  # [b,sl,ei]
        h2 = tf.multiply(inputs2, attr_2)  # [b,sl,ei]

        h = h1 + h2

        return h


class BilinearInteraction(Layer):
    def __init__(self, bilinear_type="interaction", seed=1024, **kwargs):
        self.bilinear_type = bilinear_type
        self.seed = seed

        super(BilinearInteraction, self).__init__(**kwargs)

    def build(self, input_shape):

        if not isinstance(input_shape, list) or len(input_shape) < 2:
            raise ValueError('A `AttentionalFM` layer should be called '
                             'on a list of at least 2 inputs')
        embedding_size = int(input_shape[0][-1])

        if self.bilinear_type == "all":
            self.W = self.add_weight(shape=(embedding_size, embedding_size), initializer=glorot_normal(
                seed=self.seed), name="bilinear_weight")
        elif self.bilinear_type == "each":
            self.W_list = [self.add_weight(shape=(embedding_size, embedding_size), initializer=glorot_normal(
                seed=self.seed), name="bilinear_weight" + str(i)) for i in range(len(input_shape) - 1)]
        elif self.bilinear_type == "interaction":
            self.W_list = [self.add_weight(shape=(embedding_size, embedding_size), initializer=glorot_normal(
                seed=self.seed), name="bilinear_weight" + str(i) + '_' + str(j)) for i, j in
                           itertools.combinations(range(len(input_shape)), 2)]
        else:
            raise NotImplementedError

        super(BilinearInteraction, self).build(
            input_shape)  # Be sure to call this somewhere!

    def call(self, inputs, **kwargs):

        if K.ndim(inputs[0]) != 3:
            raise ValueError(
                "Unexpected inputs dimensions %d, expect to be 3 dimensions" % (K.ndim(inputs)))

        n = len(inputs)
        if self.bilinear_type == "all":
            vidots = [tf.tensordot(inputs[i], self.W, axes=(-1, 0)) for i in range(n)]
            p = [tf.multiply(vidots[i], inputs[j]) for i, j in itertools.combinations(range(n), 2)]
        elif self.bilinear_type == "each":
            vidots = [tf.tensordot(inputs[i], self.W_list[i], axes=(-1, 0)) for i in range(n - 1)]
            p = [tf.multiply(vidots[i], inputs[j]) for i, j in itertools.combinations(range(n), 2)]
        elif self.bilinear_type == "interaction":
            p = [tf.multiply(tf.tensordot(v[0], w, axes=(-1, 0)), v[1])
                 for v, w in zip(itertools.combinations(inputs, 2), self.W_list)]
        else:
            raise NotImplementedError
        output = tf.concat(p, axis=1)
        return output


# 改进的SENETLayer，考虑了局部和全局特征
class FeatureAFF(Layer):
    def __init__(self, channels=64, r=4, seed=2022, **kwargs):
        self.inter_channels = int(channels // r)
        self.seed = seed

        super(FeatureAFF, self).__init__(**kwargs)

    def build(self, input_shape):
        _, self.filed_size, self.embedding_size = input_shape
        # 全局模块
        self.W_11 = self.add_weight(shape=(self.filed_size, self.inter_channels),
                                    initializer=glorot_normal(seed=self.seed),
                                    name="W_11")
        self.W_12 = self.add_weight(shape=(self.inter_channels, self.filed_size),
                                    initializer=glorot_normal(seed=self.seed),
                                    name="W_12")

        # 局部模块
        self.W_21 = self.add_weight(shape=(self.embedding_size, self.inter_channels),
                                    initializer=glorot_normal(seed=self.seed),
                                    name="W_21")
        self.W_22 = self.add_weight(shape=(self.inter_channels, 1),
                                    initializer=glorot_normal(seed=self.seed),
                                    name="W_22")

        self.tensordot = Lambda(lambda x: tf.tensordot(x[0], x[1], axes=(-1, 0)))

        super(FeatureAFF, self).build(input_shape)

    def call(self, inputs, **kwargs):
        Z = reduce_mean(inputs, axis=-1)  # [b,sl] 全局
        A_11 = tf.nn.relu(self.tensordot([Z, self.W_11]))  # [b,inter_channels]
        A_12 = tf.nn.relu(self.tensordot([A_11, self.W_12]))  # [b,sl]
        V_G = tf.multiply(inputs, tf.expand_dims(A_12, axis=2))  # [b,sl,ei]

        A_21 = tf.nn.relu(self.tensordot([inputs, self.W_21]))  # [b,sl,inter_channels]
        A_22 = tf.nn.relu(self.tensordot([A_21, self.W_22]))  # [b,sl,1]
        V_L = tf.multiply(inputs, A_22)  # [b,sl,ei]

        V = inputs * tf.nn.sigmoid(V_G + V_L) + inputs

        return V  # [b,sl,ei]


# 特征交互层
class InteractionCross(Layer):
    def __init__(self, layer_num=3, l2_reg=0, seed=1024, **kwargs):
        self.layer_num = layer_num
        self.l2_reg = l2_reg
        self.seed = seed
        super(InteractionCross, self).__init__(**kwargs)

    def build(self, input_shape):
        if len(input_shape) != 2:
            raise ValueError(
                "Unexpected inputs dimensions %d, expect to be 2 dimensions" % (len(input_shape),))
        dim = int(input_shape[-1])

        self.w = self.add_weight(name='kernel_w', shape=(dim, dim),
                                 initializer=glorot_normal(seed=self.seed))

        self.b = self.add_weight(name='bias_w', shape=(dim,),
                                 initializer=glorot_normal(seed=self.seed))

        self.weight = [self.add_weight(name='kernel' + str(i),
                                       shape=(dim, dim),
                                       initializer=glorot_normal(
                                           seed=self.seed),
                                       regularizer=l2(self.l2_reg),
                                       trainable=True) for i in range(self.layer_num)]
        self.bias = [self.add_weight(name='bias' + str(i),
                                     shape=(dim,),
                                     initializer=Zeros(),
                                     trainable=True) for i in range(self.layer_num)]

        self.projection_h = self.add_weight(shape=(dim, 1),
                                            initializer=glorot_normal(seed=self.seed), name="projection_h")
        self.projection_p = self.add_weight(shape=(dim, 1), initializer=glorot_normal(seed=self.seed),
                                            name="projection_p")

        self.tensordot = Lambda(lambda x: tf.tensordot(x[0], x[1], axes=(-1, 0)))

        super(InteractionCross, self).build(input_shape)

    def call(self, inputs, **kwargs):  # [b,ei]
        if K.ndim(inputs) != 2:
            raise ValueError(
                "Unexpected inputs dimensions %d, expect to be 2 dimensions" % (K.ndim(inputs)))
        h0 = inputs
        hi = tf.nn.relu(self.tensordot([h0, self.w]) + self.b)

        # 分层交互
        for i in range(self.layer_num):
            hi = tf.nn.relu(tf.nn.bias_add(tf.tensordot(hi, self.weight[i], axes=(-1, 0)), self.bias[i]))

        attr_score = tf.nn.softmax(tf.tensordot(hi, self.projection_h, axes=(-1, 0)), dim=1)  # [b,1]
        h = tf.multiply(hi, attr_score)  # [b,ei]
        return h


class FM(Layer):
    def __init__(self, **kwargs):

        super(FM, self).__init__(**kwargs)

    def build(self, input_shape):
        if len(input_shape) != 3:
            raise ValueError("Unexpected inputs dimensions % d,\
                             expect to be 3 dimensions" % (len(input_shape)))

        super(FM, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs, **kwargs):

        if K.ndim(inputs) != 3:
            raise ValueError(
                "Unexpected inputs dimensions %d, expect to be 3 dimensions"
                % (K.ndim(inputs)))

        concated_embeds_value = inputs

        square_of_sum = tf.square(reduce_sum(concated_embeds_value, axis=1, keep_dims=True))
        sum_of_square = reduce_sum(concated_embeds_value * concated_embeds_value, axis=1, keep_dims=True)
        cross_term = square_of_sum - sum_of_square
        cross_term = 0.5 * reduce_sum(cross_term, axis=2, keep_dims=False)

        return cross_term

    def compute_output_shape(self, input_shape):
        return (None, 1)
