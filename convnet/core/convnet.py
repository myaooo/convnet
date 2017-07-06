#
# file: convnet.py
# author: MING Yao
#

import os

import numpy as np
import tensorflow as tf

from convnet.utils.io_utils import before_save, get_path
from convnet.core.preprocess import DataGenerator, ImageDataGenerator
from convnet.core.sequential_net import SequentialNet, Layer
from convnet.core.config import data_type, Int32, save_keys
from convnet.core.utils import get_loss_func, config_proto
from convnet.core.layers import InputLayer, ConvLayer, FullyConnectedLayer, BatchNormLayer, \
    DropoutLayer, PoolLayer, AugmentLayer, FlattenLayer, ResLayer, ResBottleNeckLayer

SEED = None


def top_k_acc(predictions, labels, k=1, name=None):
    in_k = tf.nn.in_top_k(predictions, labels, k)
    return tf.reduce_mean(tf.to_float(in_k), name=name)


def in_top_k(predictions, labels, k=1, name=None):
    return tf.nn.in_top_k(predictions, labels, k, name)


def error_rate(predictions, labels):
    """Return the error rate based on dense predictions and sparse labels."""
    return 100.0 - (100.0 * np.sum(np.argmax(predictions, 1) == labels) / predictions.shape[0])


class ConvNet(SequentialNet):
    """A wrapper class for conv net in tensorflow"""

    def __init__(self, name='convnet', dtype=data_type()):
        super(ConvNet, self).__init__(name)
        self.dtype = dtype
        # array of layer names
        self.layer_names = []
        self._sess = None
        self.models = {}

        self._loss_func = None

        self.graph = tf.Graph()
        self._logdir = None
        self._saver = None
        self.finalized = False
        self._init_op = None

    @property
    def loss_func(self):
        return self._loss_func

    @loss_func.setter
    def loss_func(self, new_loss_func):
        self._loss_func = get_loss_func(new_loss_func)

    def push_back(self, layer):
        assert isinstance(layer, Layer), "layer must be instance of Layer, but got " + str(type(layer))
        layer_name = str(layer.__class__.__name__) + str(self.size)
        self.layer_names.append(layer_name)
        layer.name = layer_name
        super(ConvNet, self).push_back(layer)

    def push_input_layer(self, dshape=None):
        """
        Push a input layer. should be the first layer to be pushed.
        :param dshape: (batch_size, size_x, size_y, num_channel)
        :return: None
        """
        self.push_back(InputLayer(dshape))

    def push_augment_layer(self, padding=0, crop=0, horizontal_flip=False, per_image_standardize=False):
        self.push_back(AugmentLayer(padding, crop, horizontal_flip, per_image_standardize))

    def push_conv_layer(self, filter_size, out_channels, strides, padding='SAME', activation='linear', has_bias=False):
        self.push_back(ConvLayer(filter_size, out_channels, strides, padding, activation, has_bias))

    def push_pool_layer(self, typename, kernel_size, strides, padding='SAME'):
        self.push_back(PoolLayer(typename, kernel_size, strides, padding))

    def push_fully_connected_layer(self, out_channels, activation='linear', has_bias=True):
        self.push_back(FullyConnectedLayer(out_channels, activation, has_bias))

    def push_flatten_layer(self):
        self.push_back(FlattenLayer())

    def push_dropout_layer(self, keep_prob):
        self.push_back(DropoutLayer(keep_prob))

    def push_batch_norm_layer(self, decay=0.9, epsilon=0.001, activation='linear'):
        self.push_back(BatchNormLayer(decay=decay, epsilon=epsilon, activation=activation))

    def push_res_layer(self, filter_size, out_channels, strides, padding='SAME', activation='relu',
                       activate_before_residual=True, decay=0.9, epsilon=0.001):
        self.push_back(ResLayer(filter_size, out_channels, strides, padding, activation=activation,
                                activate_before_residual=activate_before_residual, decay=decay, epsilon=epsilon))

    def push_res_bn_layer(self, filter_size, out_channels, strides, padding='SAME', activation='relu',
                          activate_before_residual=True, decay=0.9, epsilon=0.001):
        self.push_back(ResBottleNeckLayer(filter_size, out_channels, strides, padding, activation=activation,
                                          activate_before_residual=activate_before_residual, decay=decay,
                                          epsilon=epsilon))

    def compile(self, train=True, eval=True):
        """
        Compile the model. Call before training or evaluating
        :param train: flag determine whether to compile train nodes
        :param eval: flag determine whether to compile eval nodes
        :return: None
        """
        print('compiling ' + self.name + ' model')
        with self.graph.as_default():
            with tf.variable_scope(self.name):
                super().compile()
                if train:
                    self.models['train'] = ConvModel(self, True, name='train')
                if eval:
                    self.models['eval'] = ConvModel(self, False, name='eval')

    def eval(self, data, batch_size, keys=None):
        """
        The evaluating function that will be called at the training API
        :param data: a tuple (X, Y)
        :param keys: the target metrics, e.g.: ['loss', 'acc']
        :return: the required keys
        """
        data_generator = ImageDataGenerator(data[0], data[1], batch_size, epoch_num=1, shuffle=False)
        return self.run_with_context(self.models['eval'].eval, data_generator, keys)

    def infer(self, data_generator=None, data=None, batch_size=64):
        return self.run_with_context(self.models['eval'].infer, data_generator, data, batch_size)

    @property
    def logdir(self):
        return self._logdir or get_path('./models', self.name)

    def save(self, path=None):
        """
        Save the trained model to disk
        :param path: path to save the graph and variables
        :return: None
        """
        self.finalize()
        path = path if path is not None else os.path.join(self.logdir, 'model')
        before_save(path)
        self._saver.save(self.sess, path)
        print("Model variables saved to {}.".format(get_path(path, absolute=True)))

    def restore(self, path=None):
        self.finalize()
        path = path if path is not None else self.logdir
        checkpoint = tf.train.latest_checkpoint(path)
        self._saver.restore(self.sess, checkpoint)
        print("Model variables restored from {}.".format(get_path(path, absolute=True)))

    @property
    def sess(self):
        self.finalize()
        if self._sess is None or self._sess._closed:
            self._sess = tf.Session(graph=self.graph, config=config_proto())
            self._sess.run(self._init_op)
        return self._sess

    def finalize(self):
        """
        After all the computation ops are built in the graph, build a supervisor which implicitly finalize the graph
        :return: False if the model has already been finalized
        """
        if self.finalized:
            return False
        with self.graph.as_default():
            self._init_op = tf.global_variables_initializer()
            variables = []
            for key in save_keys:
                variables += tf.get_collection(key)
            self._saver = tf.train.Saver(variables)
        self.finalized = True
        # self.graph.finalize()
        # self.supervisor = tf.train.Supervisor(self.graph, logdir=self.logdir)
        return True

    def run_with_context(self, func, *args, **kwargs):
        assert self.is_compiled
        self.finalize()
        with self.graph.as_default():
            with self.sess.as_default():
                return func(self.sess, *args, **kwargs)


class ConvModel(object):
    """
    A model instance, a class that wraps the result of the compilation of ConvNet
    """
    def __init__(self, model: ConvNet, train: bool = False, batch_size: int = None, name=''):
        self._model = model
        self.train = train
        self.batch_size = batch_size
        self.name = name
        with tf.name_scope(name):
            self.data_node = tf.placeholder(data_type(), [None] + model.front.output_shape, 'data')
            self.labels_node = tf.placeholder(Int32, [None, ], 'labels')
            self.logits = model(self.data_node, train=train, name=name)
            self.prediction = tf.nn.softmax(self.logits)
            self.acc = top_k_acc(self.prediction, self.labels_node, 1, 'acc')
            self.acc3 = top_k_acc(self.prediction, self.labels_node, 3, 'acc')
            self.loss_weights = tf.placeholder(dtype=data_type(), shape=[None])
            if train:
                self.loss = tf.reduce_mean(model.loss_func(self.logits, self.labels_node, self.loss_weights))
            else:
                self.loss = tf.reduce_mean(model.loss_func(self.logits, self.labels_node))

    def run(self, sess: tf.Session, batch_data, batch_label, ops, additional_feed=None):
        feed_dict = {self.data_node: batch_data,
                     self.labels_node: batch_label}
        if additional_feed is not None:
            feed_dict.update(additional_feed)
        return sess.run(ops, feed_dict)

    def get_fetches(self, keys=None):
        if keys is None:
            keys = ['loss', 'acc']
        elif isinstance(keys, str):
            keys = [keys]
        ops = {}
        for key in keys:
            if key == 'loss':
                ops[key] = self.loss
            if key == 'acc':
                ops[key] = self.acc
            if key == 'acc3':
                ops['acc3'] = self.acc3
        return ops

    def eval(self, sess: tf.Session, data_generator: DataGenerator, keys=None):
        ops = self.get_fetches(keys)
        logs = {key: 0 for key in ops.keys()}
        for data, label in data_generator:
            _logs = self.run(sess, data, label, ops)
            for key in logs:
                logs[key] += _logs[key]
        for key in logs:
            logs[key] /= len(data_generator)
        return logs

    def infer(self, sess: tf.Session, data_generator=None, data=None, batch_size=64):
        predictions = []
        if data_generator is None:
            assert data is not None, "data_generator and data cannot both be None"
            data_generator = DataGenerator(data, batch_size, epoch_num=1)
        assert isinstance(data_generator, DataGenerator)
        for data, label in data_generator:
            prediction = self.run(sess, data, label, self.prediction)
            predictions.append(prediction)
        return np.concatenate(predictions)
