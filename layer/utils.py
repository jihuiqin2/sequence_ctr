import tensorflow as tf
from operator import mul
from functools import reduce, wraps

"""
按一定方式计算张量中元素，
axis指定按哪个维度进行加和，默认是所有元素加和
keep_dims=False表示不维持原来张量的维度
"""


def reduce_mean(input_tensor, axis=None, keep_dims=False, name=None):
    return tf.reduce_mean(input_tensor, axis=axis, keep_dims=keep_dims, name=name)


def reduce_sum(input_tensor, axis=None, keep_dims=False, name=None):
    return tf.reduce_sum(input_tensor, axis=axis, keep_dims=keep_dims, name=name)


def reduce_max(input_tensor, axis=None, keep_dims=False, name=None):
    return tf.reduce_max(input_tensor, axis=axis, keep_dims=keep_dims, name=name)


# 归一化
def layer_norm(inputs, name='layerNorm', epsilon=1e-8):
    mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
    normalized = (inputs - mean) / (tf.sqrt(variance + epsilon))

    params_shape = inputs.get_shape()[-1:]
    gamma = tf.get_variable(name + 'gamma', params_shape, tf.float32, tf.ones_initializer())
    beta = tf.get_variable(name + 'beta', params_shape, tf.float32, tf.zeros_initializer())

    outputs = gamma * normalized + beta
    return outputs


# dice激活函数
def dice(_x, axis=-1, epsilon=0.000000001, name='dice'):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        alphas = tf.get_variable('alpha' + name, _x.get_shape()[-1],
                                 initializer=tf.constant_initializer(0.0),
                                 dtype=tf.float32)
        input_shape = list(_x.get_shape())

        reduction_axes = list(range(len(input_shape)))
        del reduction_axes[axis]
        broadcast_shape = [1] * len(input_shape)
        broadcast_shape[axis] = input_shape[axis]

    # case: train mode (uses stats of the current batch)
    mean = tf.reduce_mean(_x, axis=reduction_axes)
    brod_mean = tf.reshape(mean, broadcast_shape)
    std = tf.reduce_mean(tf.square(_x - brod_mean) + epsilon, axis=reduction_axes)
    std = tf.sqrt(std)
    brod_std = tf.reshape(std, broadcast_shape)
    x_normed = (_x - brod_mean) / (brod_std + epsilon)

    x_p = tf.sigmoid(x_normed)
    return alphas * (1.0 - x_p) * _x + x_p * _x


# prelu激活函数
def prelu(_x, scope=''):
    """parametric ReLU activation"""
    with tf.variable_scope(name_or_scope=scope, default_name="prelu"):
        _alpha = tf.get_variable("prelu_" + scope, shape=_x.get_shape()[-1],
                                 dtype=_x.dtype, initializer=tf.constant_initializer(0.1))
        return tf.maximum(0.0, _x) + _alpha * tf.minimum(0.0, _x)


def softmax(logits, dim=-1, name=None):
    return tf.nn.softmax(logits, dim=dim, name=name)


"""
1】
BatchNorm和LayerNorm的区别
https://blog.csdn.net/qq_31878083/article/details/121466420
"""


def default(val, default_val):
    return default_val if val is None else val


def cache_method_decorator(cache_attr, cache_namespace, reexecute=False):
    def inner_fn(fn):
        @wraps(fn)
        def wrapper(self, *args, key_namespace=None, fetch=False, set_cache=True, **kwargs):
            namespace_str = str(default(key_namespace, ''))
            _cache = getattr(self, cache_attr)
            _keyname = f'{cache_namespace}:{namespace_str}'

            if fetch:
                val = _cache[_keyname]
                if reexecute:
                    fn(self, *args, **kwargs)
            else:
                val = fn(self, *args, **kwargs)
                if set_cache:
                    setattr(self, cache_attr, {**_cache, **{_keyname: val}})
            return val

        return wrapper

    return inner_fn


def get_padding(x, padding_value=0, dtype=tf.float32):
    """Return float tensor representing the padding values in x.
    Args:
    x: int tensor with any shape
    padding_value: int value that
    dtype: The dtype of the return value.
    Returns:
    float tensor with same shape as x containing values 0 or 1.
    0 -> non-padding, 1 -> padding
    """
    return tf.cast(tf.equal(x, padding_value), dtype)


def make_unit_length(x, epsilon=1e-6):
    norm = tf.norm(x, ord=2, axis=-1, keep_dims=True)
    return tf.truediv(x, norm + epsilon)


def sort_key_val(t1, t2, axis=-1):
    # values = tf.sort(t1, axis=axis)
    dim0, dim1 = t1.get_shape().as_list()
    values, indices = tf.nn.top_k(t1, dim1)
    values = tf.reverse(values, axis=[-1])  # 改为从小到大排序

    offset = tf.range(tf.shape(t1)[0]) * dim1
    offset = tf.reshape(offset, [-1, 1])  # [?,1]

    offset = tf.tile(offset, [1, dim1])  # [?,dim1]
    t2 = tf.tile(t2, [tf.shape(t1)[0], 1])  # [?,dim1]

    _, indices = tf.nn.top_k(t1, dim1)  # 从大到小排序
    indices = tf.reverse(indices, axis=[-1])  # 改为从小到大排序
    return values, tf.gather(tf.reshape(t2, [-1]), indices + offset, axis=axis)


def batched_index_select(values, indices):
    dim0, dim1 = indices.get_shape().as_list()
    seq_len = values.shape[1]
    last_dim = values.shape[-1]

    offset = tf.range(tf.shape(indices)[0]) * seq_len
    offset = tf.reshape(offset, [-1, 1])  # [?,1]
    offset = tf.tile(offset, [1, dim1])  # [?,dim1]
    # offset = tf.broadcast_to(offset, indices.shape)

    flatten_values = tf.reshape(values, [-1, last_dim])
    return tf.gather(flatten_values, indices + offset)


def process_inputs_chunk(fn, *args, seed_=None, chunks=1):
    chunked_inputs = list(map(lambda x: tf.split(x, chunks, axis=-2), args))
    outputs = [fn(*input_pair, seed_) for i, input_pair in
               enumerate(zip(*chunked_inputs))]  # chunking 된 q ,kv 끼리 묶여서 input_pair를 만든다.
    return outputs


def cache_fn(f):
    cache = None

    def cached_fn(*args, **kwargs):
        nonlocal cache
        if cache is not None:
            return cache
        cache = f(*args, **kwargs)
        return cache

    return cached_fn


def merge_dims(ind_from, ind_to, tensor):
    shape = tensor.get_shape().as_list()
    arr_slice = slice(ind_from, ind_to + 1)
    shape[arr_slice] = [reduce(mul, shape[arr_slice])]
    return tf.reshape(tensor, tuple(shape))


def split_dims(ind_from, ind_to, tensor):
    shape = list(tensor.shape)
    arr_slice = slice(ind_from, ind_to + 1)
    shape[arr_slice] = [reduce(mul, shape[arr_slice])]
    return tf.reshape(tensor, tuple(shape))


class TriangularCausalMask():
    def __init__(self, B, L):
        mask_shape = [B, 1, L, L]

        mask_a = tf.linalg.band_part(tf.ones(mask_shape), 0, -1)  # Upper triangular matrix of 0s and 1s
        mask_b = tf.linalg.band_part(tf.ones(mask_shape), 0, 0)  # Diagonal matrix of 0s and 1s
        mask = tf.cast(mask_a - mask_b, dtype=tf.float32)

        self._mask = mask
        tf.stop_gradient(self._mask)

    @property
    def mask(self):
        return self._mask


class ProbMask():
    def __init__(self, B, H, L, index, scores):
        _mask = tf.ones((L, scores.shape[-1]))

        mask_a = tf.linalg.band_part(_mask, 0, -1)  # Upper triangular matrix of 0s and 1s
        mask_b = tf.linalg.band_part(_mask, 0, 0)  # Diagonal matrix of 0s and 1s
        _mask = tf.cast(mask_a - mask_b, dtype=tf.float32)

        _mask_ex = tf.broadcast_to(_mask, [B, H, L, scores.shape[-1]])
        indicator = _mask_ex[tf.range(B)[:, None, None],
                    tf.range(H)[None, :, None],
                    index, :]
        self._mask = indicator.reshape(scores.shape)

    @property
    def mask(self):
        return self._mask
