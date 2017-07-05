import numpy as np
import tensorflow as tf

from convnet.core.sequential_net import SequentialNet, Layer
from convnet.core.utils import get_activation, output_shape
from convnet.core.config import data_type, weight_keys, bias_keys, local_keys, Float32


class InputLayer(Layer):
    """Input Layer"""

    def __init__(self, dshape):
        """

        :param dshape: shape of input data. [batch_size, size_x, size_y, num_channel]
        """
        super().__init__('input', 'input')
        self.dshape = dshape

    def __call__(self, input_, train=False, name=''):
        super().__call__(input_)
        return input_

    def compile(self):
        return

    @property
    def output_shape(self):
        return list(self.dshape[1:])

    @property
    def is_compiled(self):
        """
        No need to compile
        """
        return True


class AugmentLayer(Layer):
    def __init__(self, padding=0, crop=0, horizontal_flip=False, per_image_standardize=False, name=''):
        super().__init__("augment", name)
        self.padding = padding
        self.crop = crop
        self.horizontal_flip = horizontal_flip
        self.per_image_standardize = per_image_standardize
        self._shape = None

    def __call__(self, input_, train=False, name=''):
        prev_shape = self.prev.output_shape
        if train:
            if self.padding > 0:
                input_ = tf.map_fn(lambda img: tf.image.resize_image_with_crop_or_pad(
                    img, prev_shape[0] + self.padding, prev_shape[1] + self.padding), input_)
            if self.crop > 0:
                input_ = tf.map_fn(lambda img: tf.random_crop(img, self.output_shape), input_)
            if self.horizontal_flip:
                input_ = tf.map_fn(lambda img: tf.image.random_flip_left_right(img), input_)
        # Brightness/saturation/constrast provides small gains .2%~.5% on cifar.
        # image = tf.image.random_brightness(image, max_delta=63. / 255.)
        # image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        # image = tf.image.random_contrast(image, lower=0.2, upper=1.8)
        else:
            diff = self.padding - self.crop
            if diff != 0:
                input_ = tf.image.resize_image_with_crop_or_pad(input_, self.output_shape[0], self.output_shape[1])
        if self.per_image_standardize:
            input_ = tf.map_fn(tf.image.per_image_standardization, input_)
        return input_

    def compile(self):
        pass

    @property
    def output_shape(self):
        if self._shape is None:
            prev_shape = self.prev.output_shape
            diff = self.padding - self.crop
            self._shape = [prev_shape[0] + diff, prev_shape[1] + diff, prev_shape[2]]
        return self._shape

    @property
    def is_compiled(self):
        """
        No need to compile
        """
        return True


class ConvLayer(Layer):
    """Input Layer"""

    def __init__(self, filter_size, out_channels, strides, padding='SAME', activation='linear',
                 has_bias=True, dtype=data_type(), name=''):
        """

        :param filter_size: of shape [m,n]
        :param out_channels:  output channel number, should be an integer
        :param strides: stride size, of shape [x,y]
        :param name:
        :param padding: 'VALID' or 'SAME'
        :param activation:
        :param has_bias:
        """
        super().__init__('conv', name)
        self._filter_shape = filter_size
        self._out_channels = out_channels
        if len(strides) == 2:
            strides = [1] + strides + [1]
        assert len(strides) == 4
        self.strides = strides
        self.padding = padding
        self.activation = get_activation(activation)
        self.has_bias = has_bias
        self.filters = None
        self.bias = None
        self.shape = None
        self.dtype = dtype
        self._is_compiled = False
        self._output_shape = None

    def __call__(self, input_, train=True, name=''):
        with tf.variable_scope(self.name, reuse=True):
            super().__call__(input_)
            results = tf.nn.conv2d(input_, self.filters, self.strides, self.padding, name='conv' + name)
            if self.has_bias:
                results = results + self.bias
            return self.activation(results)

    def compile(self):
        assert self.prev is not None
        input_shape = self.prev.output_shape
        # Compute weight shapes
        in_channels = input_shape[-1]
        out_channels = self._out_channels
        self.shape = self._filter_shape + [in_channels, out_channels]

        # Initialize Variables
        with tf.variable_scope(self.name) as scope:
            n = self.shape[0] * self.shape[1] * out_channels
            self.filters = tf.get_variable('filters', shape=self.shape, dtype=self.dtype,
                                           initializer=tf.random_normal_initializer(
                                               stddev=np.sqrt(2.0 / n), dtype=self.dtype
                                           ),
                                           collections=weight_keys)
            tf.summary.tensor_summary(tf.get_variable_scope().name + '_filters', self.filters)
            if self.has_bias:
                self.bias = tf.get_variable('bias', shape=[out_channels], dtype=self.dtype,
                                            initializer=tf.constant_initializer(0.01, dtype=self.dtype),
                                            collections=bias_keys)
                tf.summary.tensor_summary(tf.get_variable_scope().name + '_bias', self.bias)
        self._is_compiled = True

    @property
    def output_shape(self):
        if self._output_shape is None:
            assert self.prev is not None
            input_shape = self.prev.output_shape
            x, y = output_shape(input_shape, self._filter_shape, self.strides, self.padding)
            self._output_shape = [x, y, self._out_channels]
        return self._output_shape

    @property
    def is_compiled(self):
        return self._is_compiled


class PoolLayer(Layer):
    """Pooling Layer"""

    def __init__(self, typename, filter_shape, strides, padding='SAME', name=''):
        if typename == 'max':
            super().__init__('max_pool')
            self.pool_func = tf.nn.max_pool
        elif typename == 'avg':
            super().__init__('avg_pool')
            self.pool_func = tf.nn.avg_pool
        if len(filter_shape) == 2:
            filter_shape = [1] + filter_shape + [1]
        self._filter_shape = filter_shape
        self._output_shape = None
        if len(strides) == 2:
            strides = [1] + strides + [1]
        self.strides = strides
        self.padding = padding
        self.name = name

    def __call__(self, input_, train=True, name=''):
        with tf.variable_scope(self.name, reuse=True):
            super().__call__(input_)
            return self.pool_func(input_, self._filter_shape, self.strides, self.padding, name=self.type + name)

    def compile(self):
        return

    @property
    def output_shape(self):
        if self._output_shape is None:
            assert self.prev is not None
            input_shape = self.prev.output_shape
            x, y = output_shape(input_shape, self._filter_shape, self.strides, self.padding)
            self._output_shape = [x, y, input_shape[-1]]
        return self._output_shape

    @property
    def is_compiled(self):
        return True


class DropoutLayer(Layer):
    def __init__(self, keep_prob, name=''):
        super().__init__('dropout', name)
        self.keep_prob = keep_prob
        self._output_shape = None

    def __call__(self, input_, train=True, name=''):
        if not train:
            return input_
        with tf.variable_scope(self.name, reuse=True):
            super().__call__(input_)
            keep_prob = tf.constant(self.keep_prob, dtype=Float32, name='keep_prob')
            return tf.nn.dropout(input_, keep_prob, name='dropout' + name)

    def compile(self):
        return

    @property
    def output_shape(self):
        if self._output_shape is None:
            self._output_shape = self.prev.output_shape
        return self._output_shape

    @property
    def is_compiled(self):
        return True


class FlattenLayer(Layer):
    def __init__(self, name=''):
        super().__init__('flatten', name)
        self._output_shape = None

    def __call__(self, input_, train=True, name=''):
        with tf.variable_scope(self.name, reuse=True):
            super(FlattenLayer, self).__call__(input_)
            shape = input_.get_shape().as_list()
            shape0 = shape[0] if shape[0] is not None else -1
            return tf.reshape(input_, [shape0, shape[1] * shape[2] * shape[3]], name='flatten' + name)

    def compile(self):
        return

    @property
    def output_shape(self):
        if self._output_shape is None:
            input_shape = self.prev.output_shape
            self._output_shape = [input_shape[0] * input_shape[1] * input_shape[2]]
        return self._output_shape

    @property
    def is_compiled(self):
        return True


class PadLayer(Layer):
    def __init__(self, paddings, name=''):
        super().__init__('pad', name)
        self._output_shape = None
        self.paddings = paddings

    def __call__(self, input_, train=True, name=''):
        with tf.variable_scope(self.name, reuse=True):
            super(PadLayer, self).__call__(input_)
            return tf.pad(input_, self.paddings)

    def compile(self):
        return

    @property
    def output_shape(self):
        if self._output_shape is None:
            input_shape = self.prev.output_shape
            self._output_shape = []
            for i, shape in enumerate(input_shape):
                self._output_shape.append(shape if shape is None else shape + sum(self.paddings[i]))
        return self._output_shape

    @property
    def is_compiled(self):
        return True


class FullyConnectedLayer(Layer):
    def __init__(self, out_channels, activation='linear', has_bias=True, dtype=data_type(), name=''):
        super().__init__('fully_connected', name)
        self.activation = get_activation(activation)
        self.has_bias = has_bias
        self.shape = None
        self._out_channels = out_channels
        self.dtype = dtype
        self._is_compiled = False
        self.weights = None
        self.bias = None

    def __call__(self, input_, train=True, name=''):
        with tf.variable_scope(self.name, reuse=True):
            input_ = super().__call__(input_)
            result = tf.matmul(input_, self.weights)
            if self.has_bias:
                result = result + self.bias
            return self.activation(result)

    def compile(self):
        assert self.prev is not None
        input_shape = self.prev.output_shape
        # Compute weight shapes
        in_channels = input_shape[-1]
        out_channels = self._out_channels
        self.shape = [in_channels, out_channels]
        # Initialize Variables
        with tf.variable_scope(self.name) as scope:
            self.weights = tf.get_variable('weights', shape=self.shape, dtype=self.dtype,
                                           initializer=tf.uniform_unit_scaling_initializer(factor=1.0,
                                                                                           dtype=self.dtype),
                                           collections=weight_keys)
            tf.summary.tensor_summary(tf.get_variable_scope().name + '_weights', self.weights)
            if self.has_bias:
                self.bias = tf.get_variable('bias', shape=[out_channels], dtype=self.dtype,
                                            initializer=tf.constant_initializer(0.01, dtype=self.dtype),
                                            collections=bias_keys)
            tf.summary.tensor_summary(tf.get_variable_scope().name + '_bias', self.bias)
        self._is_compiled = True

    @property
    def output_shape(self):
        return [self._out_channels]

    @property
    def is_compiled(self):
        return self._is_compiled


class BatchNormLayer(Layer):
    def __init__(self, decay=0.9, epsilon=0.001, activation='linear', name=''):
        super().__init__('batch_norm', name)
        self._output_shape = None
        self.decay = decay
        self.epsilon = epsilon
        self.activation = activation
        self.mode = None

    def compile(self):
        return True

    @property
    def output_shape(self):
        if self._output_shape is None:
            self._output_shape = self.prev.output_shape
        return self._output_shape

    def __call__(self, input_, train=True, name=''):
        super().__call__(input, train)
        reuse = True if self.n_calls > 1 else None
        with tf.variable_scope(self.name, reuse=reuse):
            results = tf.contrib.layers.batch_norm(input_, decay=self.decay, epsilon=self.epsilon, scale=True,
                                                   is_training=train, reuse=reuse, scope='bn_op',
                                                   variables_collections=local_keys)
            return get_activation(self.activation)(results)

    @property
    def is_compiled(self):
        return True


class ResLayer(Layer):
    def __init__(self, filter_size, out_channels, strides, padding='SAME', activation='relu',
                 activate_before_residual=True, decay=0.9, epsilon=0.001, name=''):
        super().__init__('res', name)
        self._filter_size = filter_size
        self._out_channels = out_channels
        self.strides = strides
        self.padding = padding
        self.activation = get_activation(activation)
        self.activate_before_residual = activate_before_residual
        self.decay = decay
        self.epsilon = epsilon
        self.filters = None
        self.bias = None
        self.shape = None
        self._is_compiled = False
        self._output_shape = None
        self.net1 = SequentialNet(name)
        self.net2 = SequentialNet(name + '_residue')

    def __call__(self, input_, train=True, name=''):
        assert self.is_compiled
        with tf.variable_scope(self.name):
            results = self.net1(input_, train, 'pipe')
            res = self.net2(input_, train, 'res')
            return res + results

    def compile(self):
        input_shape = self.prev.output_shape
        # Compute weight shapes
        in_channels = input_shape[-1]
        out_channels = self._out_channels
        # self.shape = self._filter_size + [in_channels, out_channels]
        assert len(self.net1) == 0
        self.net1.append(InputLayer(dshape=[None] + input_shape))
        self.net2.append(InputLayer(dshape=[None] + input_shape))
        if self.activate_before_residual:
            self.net1.append(BatchNormLayer(decay=self.decay, epsilon=self.epsilon,
                                            activation=self.activation, name=self.net1.name + '_bn0'))
        self.net1.append(ConvLayer(self._filter_size, out_channels, self.strides,
                                   self.net1.name + '_conv1', activation='linear',
                                   has_bias=False))
        self.net1.append(BatchNormLayer(decay=self.decay, epsilon=self.epsilon,
                                        activation=self.activation, name=self.net1.name + 'bn1'))
        self.net1.append(ConvLayer(self._filter_size, out_channels, [1, 1],
                                   activation='linear', has_bias=False,
                                   name=self.net1.name + '_conv2'))

        if in_channels != out_channels:
            self.net2.append(ConvLayer(self.strides, out_channels, self.strides,
                                       activation='linear', has_bias=False,
                                       name=self.net2.name + 'shortcut'))

        for net in [self.net1, self.net2]:
            net.compile()
        self._is_compiled = True

    @property
    def output_shape(self):
        assert len(self.net1) > 0 and self.net1.is_compiled
        return self.net1.back.output_shape

    @property
    def is_compiled(self):
        return self._is_compiled


class ResBottleNeckLayer(ResLayer):
    def __init__(self, filter_size, out_channels, strides, padding='SAME', activation='relu',
                 activate_before_residual=True, decay=0.9, epsilon=0.001, name=''):
        super().__init__(filter_size, out_channels, strides, padding, activation,
                         activate_before_residual, decay, epsilon, name=name)

    def compile(self):
        input_shape = self.prev.output_shape
        # Compute weight shapes
        in_channels = input_shape[-1]
        out_channels = self._out_channels
        # self.shape = self._filter_size + [in_channels, out_channels]
        assert len(self.net1) == 0
        self.net1.append(InputLayer(dshape=[None] + input_shape))
        self.net2.append(InputLayer(dshape=[None] + input_shape))
        if self.activate_before_residual:
            self.net1.append(BatchNormLayer(decay=self.decay,
                                            epsilon=self.epsilon, activation=self.activation,
                                            name=self.net1.name + '_bn0'))
        self.net1.append(ConvLayer([1, 1], out_channels / 4, self.strides,
                                   activation='linear', has_bias=False,
                                   name=self.net1.name + '_conv1'))
        self.net1.append(BatchNormLayer(decay=self.decay,
                                        epsilon=self.epsilon, activation=self.activation,
                                        name=self.net1.name + 'bn1'))
        self.net1.append(ConvLayer(self._filter_size, out_channels / 4, [1, 1],
                                   activation='linear', has_bias=False,
                                   name=self.net1.name + '_conv2'))
        self.net1.append(BatchNormLayer(decay=self.decay, epsilon=self.epsilon,
                                        activation=self.activation, name=self.net1.name + 'bn2'))
        self.net1.append(ConvLayer([1, 1], out_channels, [1, 1], activation='linear',
                                   has_bias=False, name=self.net1.name + '_conv3'))
        if in_channels != out_channels:
            self.net2.append(ConvLayer(self.strides, out_channels, self.strides, activation='linear',
                                       has_bias=False, name=self.net2.name + 'shortcut'))
        for net in [self.net1, self.net2]:
            net.compile()
        self._is_compiled = True
