import tensorflow as tf
from tensorflow.python.ops.init_ops import Zeros
from tensorflow.python.keras.layers import Layer, Activation, BatchNormalization


class Dice(Layer):

    def __init__(self, axis=-1, epsilon=1e-9, **kwargs):
        self.axis = axis
        self.epsilon = epsilon
        super(Dice, self).__init__(**kwargs)

    def build(self, input_shape):
        self.bn = BatchNormalization(
            axis=self.axis, epsilon=self.epsilon, center=False, scale=False)
        self.alphas = self.add_weight(shape=(input_shape[-1],), initializer=Zeros(
        ), dtype=tf.float32, name='dice_alpha')  # name='alpha_'+self.name
        super(Dice, self).build(input_shape)  # Be sure to call this somewhere!
        self.uses_learning_phase = True

    def call(self, inputs, training=None, **kwargs):
        inputs_normed = self.bn(inputs, training=training)
        # tf.layers.batch_normalization(
        # inputs, axis=self.axis, epsilon=self.epsilon, center=False, scale=False)
        x_p = tf.sigmoid(inputs_normed)
        return self.alphas * (1.0 - x_p) * inputs + x_p * inputs

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self, ):
        config = {'axis': self.axis, 'epsilon': self.epsilon}
        base_config = super(Dice, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def activation_layer(activation):
    if activation in ("dice", "Dice"):
        act_layer = Dice()
    elif isinstance(activation, (str, '')):
        act_layer = Activation(activation)
    elif issubclass(activation, Layer):
        act_layer = activation()
    else:
        raise ValueError("Invalid activation,found %s.You should use a str or a Activation Layer Class." % (activation))
    return act_layer
