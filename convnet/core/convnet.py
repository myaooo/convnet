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

    def __init__(self, name='convnet', dtype=data_type(), graph=None, logdir=None):
        super(ConvNet, self).__init__(name)
        self.dtype = dtype
        # array of layer names
        self.layer_names = []
        self._sess = None
        self.models = {}

        self._loss_func = None

        self.graph = graph or tf.Graph()
        self._logdir = logdir
        self._saver = None
        self.finalized = False
        self._init_op = None
        self.trainer = None
        self.is_training = None

    @property
    def loss_func(self):
        return self._loss_func

    @loss_func.setter
    def loss_func(self, new_loss_func):
        self._loss_func = get_loss_func(new_loss_func)

    def push_back(self, layer):
        """
        Push a layer to the back of the conv net.
        :param layer: a Layer
        :return: None
        """
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

    def compile(self):
        """
        Compile the model. Call before training or evaluating
        :param train: flag determine whether to compile train nodes
        :param eval: flag determine whether to compile eval nodes
        :return: None
        """
        assert not self.is_compiled
        print('compiling ' + self.name + ' model')
        with self.graph.as_default():
            with tf.variable_scope(self.name):
                super().compile()
                # if train:
                self.is_training = tf.placeholder(tf.bool, shape=[], name='is_training')
                with tf.name_scope("model"):
                    self.data_node = tf.placeholder(data_type(), [None] + self.front.output_shape, name='data')
                    self.label_node = tf.placeholder(Int32, [None, ], name='label')
                    logits = self(self.data_node, train=self.is_training)
                    self.prediction = tf.nn.softmax(logits, name='prediction')
                    self.acc = top_k_acc(self.prediction, self.label_node, 1, name='acc')
                    self.acc3 = top_k_acc(self.prediction, self.label_node, 3, name='acc3')
                    self.loss_weights = tf.placeholder(dtype=data_type(), shape=[None], name='loss_weights')
                    self.loss = tf.reduce_mean(self.loss_func(logits, self.label_node, self.loss_weights),
                                               name='loss')
                # self.model = ConvModel(self, self.is_training, name='train').compile()

        return self

    def compile_from_meta(self, path=None, import_scope=None, input_map=None):
        assert not self.is_compiled
        print('compiling ' + self.name + ' model from meta')
        with self.graph.as_default():
            graph = self.graph
            self.restore_graph_from_meta(path, import_scope=import_scope, input_map=input_map)
            self.is_training = self.graph.get_tensor_by_name(self.name+'/is_training:0')
            prefix = self.name + '/model/'
            self.data_node = graph.get_tensor_by_name(prefix + 'data:0')
            self.label_node = graph.get_tensor_by_name(prefix + 'label:0')
            self.prediction = graph.get_tensor_by_name(prefix + 'prediction:0')
            self.acc = graph.get_tensor_by_name(prefix + 'acc:0')
            self.acc3 = graph.get_tensor_by_name(prefix + 'acc3:0')
            # if self.train:
            self.loss_weights = graph.get_tensor_by_name(prefix + 'loss_weights:0')
            self.loss = graph.get_tensor_by_name(prefix + 'loss:0')
            # train_model = ConvModel(self, self.is_training, name='train').restore_from_graph(self.graph)
            # self.model = train_model
        self.is_compiled = True

    def eval(self, data, batch_size, keys=None):
        """
        The evaluating function that will be called at the training API
        :param data: a tuple (X, Y)
        :param keys: the target metrics to be evaluated, e.g.: ['loss', 'acc']
        :return: the required keys
        """
        data_generator = ImageDataGenerator(data[0], data[1], batch_size, epoch_num=1, shuffle=False)
        return self.run_with_context(self._eval, data_generator, keys)

    def infer(self, data_generator=None, data=None, batch_size=64):
        return self.run_with_context(self._infer, data_generator, data, batch_size)

    @property
    def logdir(self):
        return self._logdir or get_path('./models', self.name)

    @property
    def name_prefix(self):
        return self.name + '/' + 'model/'

    def save(self, sess=None, path=None, global_step=None):
        """
        Save the trained model to disk
        :param path: path to save the graph and variables
        :param global_step: additional identifier for the checkpoint file
        :return: None
        """
        # self.finalize()
        path = path if path is not None else os.path.join(self.logdir, 'model')
        before_save(path)
        sess = sess or self.sess
        checkpoint = self.saver.save(sess, path, global_step)
        print("Model variables saved to {}.".format(get_path(checkpoint, absolute=True)))

    def restore_weights(self, sess=None, path=None):
        # self._finalize()
        path = path if path is not None else self.logdir
        checkpoint = tf.train.latest_checkpoint(path)
        if checkpoint is None:
            raise FileNotFoundError('Cannot find model checkpoint from "{:s}"'.format(path))
        sess = sess or self.sess
        self.saver.restore(sess, checkpoint)
        print("Model variables restored from {}.".format(get_path(checkpoint, absolute=True)))

    def restore_graph_from_meta(self, path=None, *args, **kwargs):
        path = path or self.logdir
        checkpoint = tf.train.latest_checkpoint(path)
        if checkpoint is None:
            raise FileNotFoundError('Cannot find model checkpoint from "{:s}"'.format(path))
        with self.graph.as_default():
            tf.train.import_meta_graph(checkpoint + '.meta', *args, **kwargs)
        # self._finalize()
        print("Model graph restored from {}.".format(get_path(checkpoint, absolute=True)))
        # self.finalized = True

    @property
    def sess(self):
        # self._finalize()
        if self._sess is None or self._sess._closed:
            self._sess = tf.Session(graph=self.graph, config=config_proto())
            with self.graph.as_default():
                with tf.name_scope(self.name):
                    self._init_op = tf.variables_initializer(
                        tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name + '/'), name='init')
            self._sess.run(self._init_op)
        return self._sess

    @property
    def saver(self):
        if self._saver is None:
            with self.graph.as_default():
                with tf.name_scope(self.name):
                    # variables = []
                    variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name + '/')
                    # for key in save_keys:
                    #     variables += tf.get_collection(key, scope=self.name + '/')
                    self._saver = tf.train.Saver(variables)
        return self._saver

    # def _finalize(self):
    #     """
    #     After all the ops and variables are claimed in the graph, explicitly finalize the graph
    #     :return: None
    #     """
    #     # make sure the following code will be run only once
    #     if self.finalized:
    #         return
    #     with self.graph.as_default():
    #         with tf.name_scope(self.name):
    #             self._init_op = tf.variables_initializer(
    #                 tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name + '/'), name='init')
    #             # variables = []
    #             variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name + '/')
    #             # for key in save_keys:
    #             #     variables += tf.get_collection(key, scope=self.name + '/')
    #             self._saver = tf.train.Saver(variables)
    #     self.finalized = True

    def run_with_context(self, func, *args, **kwargs):
        assert self.is_compiled
        # self.finalize()
        with self.graph.as_default():
            with self.sess.as_default():
                return func(self.sess, *args, **kwargs)

    def _assert_compiled(self):
        assert self.is_compiled, "Make sure you explicitly compile the model!"

    def run(self, sess: tf.Session, batch_data, batch_label, ops, additional_feed=None):
        feed_dict = {self.data_node: batch_data}
        if batch_label is not None:
            feed_dict[self.label_node] = batch_label
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

    def _eval(self, sess: tf.Session, data_generator: DataGenerator, keys=None):
        ops = self.get_fetches(keys)
        logs = {key: 0 for key in ops.keys()}
        for data, label in data_generator:
            _logs = self.run(sess, data, label, ops, {self.is_training: False, self.loss_weights: np.ones(label.shape)})
            for key in logs:
                logs[key] += _logs[key]
        for key in logs:
            logs[key] /= len(data_generator)
        return logs

    def _infer(self, sess: tf.Session, data_generator=None, data=None, batch_size=64):
        predictions = []
        if data_generator is None:
            assert data is not None, "data_generator and data cannot both be None"
            data_generator = DataGenerator(data, batch_size, epoch_num=1)
        assert isinstance(data_generator, DataGenerator)
        for data in data_generator:
            prediction = self.run(sess, data, None, self.prediction,
                                  {self.is_training: False, self.loss_weights: np.ones(data.shape[0])})
            predictions.append(prediction)
        return np.concatenate(predictions)

#
# class ConvModel(object):
#     """
#     A model instance, a class that wraps the result of the compilation of ConvNet
#     """
#     def __init__(self, net: ConvNet, train=False, batch_size: int = None, name=''):
#         self._net = net
#         self.train = train
#         self.batch_size = batch_size
#         self.name = name
#         self.is_compiled = False
#
#     def compile(self):
#         if self.is_compiled:
#             return self
#         with tf.name_scope(self.name):
#             self.data_node = tf.placeholder(data_type(), [None] + self._net.front.output_shape, name='data')
#             self.label_node = tf.placeholder(Int32, [None, ], name='label')
#             logits = self._net(self.data_node, train=self.train)
#             self.prediction = tf.nn.softmax(logits, name='prediction')
#             self.acc = top_k_acc(self.prediction, self.label_node, 1, name='acc')
#             self.acc3 = top_k_acc(self.prediction, self.label_node, 3, name='acc3')
#             self.loss_weights = tf.placeholder(dtype=data_type(), shape=[None], name='loss_weights')
#             self.loss = tf.reduce_mean(self._net.loss_func(logits, self.label_node, self.loss_weights),
#                                        name='loss')
#         self.is_compiled = True
#         return self
#
#     @property
#     def input_shape(self):
#         return self.data_node.shape.as_list()[1:]
#
#     @property
#     def output_shape(self):
#         return self.prediction.shape.as_list()[1:]
#
#     def restore_from_graph(self, graph):
#         if self.is_compiled:
#             return self
#         prefix = self._net.name + '/' + self.name + '/'
#         self.data_node = graph.get_tensor_by_name(prefix + 'data:0')
#         self.label_node = graph.get_tensor_by_name(prefix + 'label:0')
#         self.prediction = graph.get_tensor_by_name(prefix + 'prediction:0')
#         self.acc = graph.get_tensor_by_name(prefix + 'acc:0')
#         self.acc3 = graph.get_tensor_by_name(prefix + 'acc3:0')
#         # if self.train:
#         self.loss_weights = graph.get_tensor_by_name(prefix + 'loss_weights:0')
#         self.loss = graph.get_tensor_by_name(prefix + 'loss:0')
#         self.is_compiled = True
#         return self
#
#     def run(self, sess: tf.Session, batch_data, batch_label, ops, additional_feed=None):
#         feed_dict = {self.data_node: batch_data}
#         if batch_label is not None:
#             feed_dict[self.label_node] = batch_label
#         if additional_feed is not None:
#             feed_dict.update(additional_feed)
#         return sess.run(ops, feed_dict)
#
#     def get_fetches(self, keys=None):
#         if keys is None:
#             keys = ['loss', 'acc']
#         elif isinstance(keys, str):
#             keys = [keys]
#         ops = {}
#         for key in keys:
#             if key == 'loss':
#                 ops[key] = self.loss
#             if key == 'acc':
#                 ops[key] = self.acc
#             if key == 'acc3':
#                 ops['acc3'] = self.acc3
#         return ops
#
#     def eval(self, sess: tf.Session, data_generator: DataGenerator, keys=None):
#         ops = self.get_fetches(keys)
#         logs = {key: 0 for key in ops.keys()}
#         for data, label in data_generator:
#             _logs = self.run(sess, data, label, ops, {self.train: False, self.loss_weights: np.ones(label.shape)})
#             for key in logs:
#                 logs[key] += _logs[key]
#         for key in logs:
#             logs[key] /= len(data_generator)
#         return logs
#
#     def infer(self, sess: tf.Session, data_generator=None, data=None, batch_size=64):
#         predictions = []
#         if data_generator is None:
#             assert data is not None, "data_generator and data cannot both be None"
#             data_generator = DataGenerator(data, batch_size, epoch_num=1)
#         assert isinstance(data_generator, DataGenerator)
#         for data in data_generator:
#             prediction = self.run(sess, data, None, self.prediction,
#                                   {self.train: False, self.loss_weights: np.ones(data.shape[0])})
#             predictions.append(prediction)
#         return np.concatenate(predictions)
