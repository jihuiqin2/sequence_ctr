import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.ops.init_ops import Zeros, glorot_normal_initializer as glorot_normal

from tensorflow.python.keras.layers import Dropout, Layer, Add, BatchNormalization
from tensorflow.python.keras.regularizers import l2

from layer.utils import reduce_sum
from layer.activation import activation_layer


# 线性层
class Linear(Layer):
    def __init__(self, l2_reg=0.0, mode=0, use_bias=False, seed=1024, **kwargs):
        super(Linear, self).__init__(**kwargs)

        self.l2_reg = l2_reg
        if mode not in [0, 1, 2]:
            raise ValueError("mode must be 0,1 or 2")
        self.mode = mode
        self.use_bias = use_bias
        self.seed = seed
        self.bias = None
        self.kernel = None

    def build(self, input_shape):
        if self.use_bias:
            self.bias = self.add_weight(name='linear_bias', shape=(1,),
                                        initializer=Zeros(), trainable=True)
        if self.mode == 1:
            self.kernel = self.add_weight(
                'linear_kernel',
                shape=[int(input_shape[-1]), 1],
                initializer=glorot_normal(self.seed),
                regularizer=l2(self.l2_reg),
                trainable=True)
        elif self.mode == 2:
            self.kernel = self.add_weight(
                'linear_kernel',
                shape=[int(input_shape[1][-1]), 1],
                initializer=glorot_normal(self.seed),
                regularizer=l2(self.l2_reg),
                trainable=True)
        super(Linear, self).build(input_shape)

    def call(self, inputs, **kwargs):
        if self.mode == 0:
            sparse_input = inputs
            linear_logit = reduce_sum(sparse_input, axis=-1, keep_dims=True)
        elif self.mode == 1:
            dense_input = inputs
            fc = tf.tensordot(dense_input, self.kernel, axes=(-1, 0))
            linear_logit = fc
        else:
            sparse_input, dense_input = inputs
            fc = tf.tensordot(dense_input, self.kernel, axes=(-1, 0))  # 矩阵相乘，[None,1]
            linear_logit = reduce_sum(sparse_input, axis=-1, keep_dims=False) + fc  # [None,1]
        if self.use_bias:
            linear_logit += self.bias

        return linear_logit


# dnn层
class DNN(Layer):
    def __init__(self, hidden_units, activation='relu', l2_reg=0, dropout_rate=0, use_bn=False, output_activation=None,
                 seed=1024, **kwargs):
        self.hidden_units = hidden_units
        self.activation = activation
        self.l2_reg = l2_reg
        self.dropout_rate = dropout_rate
        self.use_bn = use_bn
        self.output_activation = output_activation
        self.seed = seed
        self.kernels = None
        self.bias = None
        self.bn_layers = None
        self.dropout_layers = None
        self.activation_layers = None

        super(DNN, self).__init__(**kwargs)

    def build(self, input_shape):
        if len(self.hidden_units) == 0:
            raise ValueError("hidden_units is empty")

        input_size = input_shape[-1]  # 输入维度
        hidden_units = [int(input_size)] + list(self.hidden_units)

        # todo?
        K.set_learning_phase(True)

        self.kernels = [self.add_weight(name='kernel' + str(i),
                                        shape=(hidden_units[i], hidden_units[i + 1]),
                                        initializer=glorot_normal(seed=self.seed),
                                        regularizer=l2(self.l2_reg),
                                        trainable=True) for i in range(len(self.hidden_units))]
        self.bias = [self.add_weight(name='bias' + str(i),
                                     shape=(self.hidden_units[i],), initializer=Zeros(),
                                     trainable=True) for i in range(len(self.hidden_units))]
        if self.use_bn:
            self.bn_layers = [BatchNormalization() for _ in range(len(self.hidden_units))]

        self.dropout_layers = [Dropout(self.dropout_rate, seed=self.seed + i) for i in
                               range(len(self.hidden_units))]

        self.activation_layers = [activation_layer(self.activation) for _ in range(len(self.hidden_units))]

        if self.output_activation:
            self.activation_layers[-1] = activation_layer(self.output_activation)

        super(DNN, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs, training=True, **kwargs):

        deep_input = inputs

        for i in range(len(self.hidden_units)):
            fc = tf.nn.bias_add(tf.tensordot(deep_input, self.kernels[i], axes=(-1, 0)), self.bias[i])

            if self.use_bn:
                fc = self.bn_layers[i](fc)

            fc = self.activation_layers[i](fc)
            fc = self.dropout_layers[i](fc)
            deep_input = fc

        return deep_input
