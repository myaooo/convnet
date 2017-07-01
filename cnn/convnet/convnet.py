#
# file: convnet.py
# author: MING Yao
#


import math
import time
import os

import numpy as np
import tensorflow as tf

from cnn.utils.io_utils import before_save, get_path
from cnn.convnet.sequential_net import SequentialNet, Layer
from cnn.convnet.classifier import Classifier
from cnn.convnet.config import data_type, Int32, save_keys
from cnn.convnet.utils import get_loss_func, config_proto
from cnn.convnet.message_protoc.log_message import create_training_log_message, \
    log_beautiful_print
from cnn.convnet.layers import InputLayer, ConvLayer, FullyConnectedLayer, BatchNormLayer, \
    DropoutLayer, PoolLayer, AugmentLayer, FlattenLayer, ResLayer, ResBottleNeckLayer

SEED = None


# _EVAL_BATCH_SIZE = 100


def error_rate(predictions, labels):
    """Return the error rate based on dense predictions and sparse labels."""
    return 100.0 - (100.0 * np.sum(np.argmax(predictions, 1) == labels) / predictions.shape[0])


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
            self.loss = tf.reduce_mean(model.loss_func(self.logits, self.labels_node))


class ConvNet(SequentialNet, Classifier):
    """A wrapper class for conv net in tensorflow"""

    def __init__(self, name='convnet', dtype=data_type()):
        super(ConvNet, self).__init__(name)
        self.dtype = dtype
        # array of layer names
        self.layer_names = []
        self._sess = None
        self.models = {}

        # Data and labels
        # self.train_data_generator = None
        # self.test_data_generator = None
        # # batch_size to be determined when training
        # self.batch_size = -1

        # placeholders for input
        # self.train_data_node = None
        # self.train_labels_node = None
        # self.eval_data_node = None
        # self.eval_labels_node = None

        # important compuation node
        # self.logits = None
        # self.loss = None
        self._loss_func = None
        # self.acc = None

        # Evaluation node
        # self.eval_loss = None
        # self.eval_logits = None
        # self.eval_prediction = None
        # self.eval_acc = None
        # self.eval_acc3 = None

        self.graph = tf.Graph()
        # Optimization node
        # self.optimizer_op = None
        # self.optimizer = None  # Default to use Momentum
        # Either a scalar (constant) or a Tensor (variable) that holds the learning rate
        # self.learning_rate = None
        # A Tensor that tracks global step
        # with self.graph.as_default():
        #     self.global_step = tf.Variable(0, dtype=Int32, name='global_step', trainable=False)
        #     # A Tensor that holds current epoch
        #     self.batch_size_node = tf.placeholder(dtype=Int32, shape=(), name='batch_size')
        #     self.train_size_node = tf.placeholder(dtype=Int32, shape=(), name='train_size')
        #     # self.lr_feed_dict = {}
        #     # self.cur_epoch = tf.Variable(0, dtype=Float, name='cur_epoch', trainable=False)
        # self.regularizer = lambda x: 0
        # self.regularized_term = None
        self._logdir = None
        self._saver = None
        # self.graph = tf.get_default_graph()
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
        layer_name = str(layer.__class__) + str(self.size)
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

        # # creating placeholder
        # self.train_data_node = tf.placeholder(self.dtype, dshape)
        # self.train_labels_node = tf.placeholder(Int32, shape=(dshape[0],))
        # self.eval_data_node = tf.placeholder(self.dtype, shape=(None, *dshape[1:]))
        # self.eval_labels_node = tf.placeholder(Int32, shape=(None,))

    def push_augment_layer(self, padding=0, crop=0, horizontal_flip=False, per_image_standardize=False):
        self.push_back(AugmentLayer(padding, crop, horizontal_flip, per_image_standardize))

    def push_conv_layer(self, filter_size, out_channels, strides, padding='SAME', activation='linear', has_bias=False):
        self.push_back(ConvLayer(filter_size, out_channels, strides, padding, activation, has_bias))

    def push_pool_layer(self, type, kernel_size, strides, padding='SAME'):
        self.push_back(PoolLayer(type, kernel_size, strides, padding))

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

    # def set_regularizer(self, regularizer=None, scale=0):
    #     """
    #     Set the loss regularizer. default to have no regularization
    #     :param regularizer: regularizer method. 'l1' or 'l2', or a list of these method
    #     :param scale: scale factor of the regularization term
    #     :return: None
    #     """
    #     if scale == 0:
    #         self.regularizer = lambda x: 0
    #         return
    #     self.regularizer = get_regularizer(regularizer, scale)

    # def set_optimizer(self, optimizer='Momentum', *args, **kwargs):
    #     """
    #     Set the Optimizer function. See convnet.utils.get_optimizer
    #     :param optimizer: a string indicating the type of the optimizer
    #     :param kwargs: optimal args of the optimizer. See TensorFlow API
    #     :return: None
    #     """
    #     assert self.learning_rate is not None and self.learning_rate != 0
    #     self.optimizer = get_optimizer(optimizer)(self.learning_rate, *args, **kwargs)

    # def set_learning_rate(self, learning_rate=0.001, update_func=None, **kwargs):
    #     """
    #     Set learning rate adjusting scheme. Wrapped from TF api.
    #     :param learning_rate: the base learning_rate
    #     :param update_func: a callable function of form updated_rate = update_func(base_rate, global_step).
    #     :return: None
    #     """
    #     if update_func is None:
    #         self.learning_rate = learning_rate
    #     else:
    #         kwargs['global_step'] = self.global_step  # * self.batch_size_node
    #         kwargs['learning_rate'] = learning_rate
    #         kwargs['decay_steps'] = self.train_size_node
    #         self.learning_rate = get_learning_rate(update_func, **kwargs)

    # def set_data(self, train_data_generator, test_data_generator=None):
    #     """
    #     Set the training data and test data
    #     :param train_data_generator:
    #     :param test_data_generator:
    #     :return:
    #     """
    #     # assert train_data_generator.n % train_data_generator.batch_size == 0
    #     self.train_data_generator = train_data_generator
    #     # if test_data_generator is not None:
    #     # assert test_data_generator.n % test_data_generator.batch_size == 0
    #     self.test_data_generator = test_data_generator

    def _cal_loss(self, logits, labels_node, name):
        # logits: the raw output of the model
        # loss: average loss, computed by specified loss_func
        loss = tf.reduce_mean(self.loss_func(logits, labels_node))
        # add a regularized term, note that it is added after taking mean on loss value
        loss = tf.add(loss, self.regularized_term, name=name)
        return loss

    def compile(self, train=True, eval=True):
        """
        Compile the model. Call before training or evaluating
        :param eval: flag determine whether to compile evaluation nodes
        :param test: flag determin whether to compile test node
        :return: None
        """
        print('compiling ' + self.name + ' model')
        with self.graph.as_default():
            with tf.variable_scope(self.name):
                super(ConvNet, self).compile()
                if train:
                    self.models['train'] = ConvModel(self, train, name='train')
                if eval:
                    self.models['eval'] = ConvModel(self, False, name='eval')
                # init input node
                # self.train_data_node = tf.placeholder(data_type(), [None] + self.front.output_shape, 'train_data')
                # self.train_labels_node = tf.placeholder(Int32, [None, ], 'train_labels')
                # self.eval_data_node = tf.placeholder(data_type(), [None] + self.front.output_shape, 'eval_data')
                # self.eval_labels_node = tf.placeholder(Int32, [None, ], 'eval_labels')
                #
                # self.regularized_term = self.regularizer(tf.get_collection(tf.GraphKeys.WEIGHTS))
                #
                # # computation node for training ops
                # self.logits = self.model(self.train_data_node, train=True)
                # self.prediction = tf.nn.softmax(self.logits)
                # self.acc = self.top_k_acc(self.prediction, self.train_labels_node, 1, name='train_acc')
                # self.loss = self._cal_loss(self.logits, self.train_labels_node, 'regularized_loss')
                #
                # tf.summary.scalar('loss', self.loss)
                # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                # with tf.control_dependencies(update_ops):
                #     with tf.name_scope('train'):
                #         # Setup optimizer ops
                #         self.optimizer_op = self.optimizer.minimize(self.loss, self.global_step)
                #
                # # Computation node for evaluations
                # if eval:
                #     with tf.name_scope('eval'):
                #         self.eval_logits = self.model(self.eval_data_node, train=False)
                #         self.eval_loss = self._cal_loss(self.eval_logits, self.eval_labels_node, 'regularized_loss')
                #         # prediction
                #         self.eval_prediction = tf.nn.softmax(self.eval_logits, name='prediction')
                #         # accuracy
                #         self.eval_acc = self.top_k_acc(self.eval_prediction, self.eval_labels_node, 1, name='acc')
                #         self.eval_acc3 = self.top_k_acc(self.eval_prediction, self.eval_labels_node, 5, name='acc3')


    # def model(self, data, train=False):
    #     """
    #     The Model definition. Feed in the data and run it through the network
    #     :param data: A Tensor that holds the data
    #     :param train: A Boolean, indicating whether the method is called by a training process or a evaluating process
    #     :return: A Tensor that holds the output of the model, or `logits`.
    #     """
    #     # Note that {strides} is a 4D array whose
    #     # shape matches the data layout: [image index, y, x, depth].
    #
    #     name = '_train' if train else '_eval'
    #     return self.__call__(data, train, name)

    # def feed_dict(self):
    #     """
    #     Evaluate global_step tensor as step and use step to generate feed_dict for **training**
    #     :return: A dict used as feed_dict in a mini-batch of training
    #     """
    #     # step = tf.train.global_step(self.sess, self.global_step) - 1
    #     # train_size = self.train_size
    #     # offset = (step * self.batch_size) % (train_size - self.batch_size)
    #     batch_data, batch_labels = self.train_data_generator.next()
    #     feed_dict = {self.train_data_node: batch_data,
    #                  self.train_labels_node: batch_labels}
    #     feed_dict.update(self.lr_feed_dict)
    #     return feed_dict

    def train(self, batch_size, num_epochs=20, eval_frequency=math.inf):
        """
        Main API for training
        :param batch_size: batch size for mini-batch
        :param num_epochs: total training epochs num
        :param eval_frequency: evaluation frequency, i.e., evaluate per `eval_freq` steps.
            Default as math.inf (No evaluation at all).
        :return: None
        """
        self.batch_size = batch_size
        train_size = self.train_size
        eval_size = self.test_size
        self.lr_feed_dict = {self.train_size_node: train_size, self.batch_size_node: batch_size}

        batch_per_epoch = int(train_size / batch_size)
        total_step = int(batch_per_epoch * num_epochs)
        eval_frequency *= batch_per_epoch
        print('start training...')
        epoch_time = step_time = start_time = time.time()
        sess = self.sess
        losses = []
        accs = []
        valid_losses = []
        valid_accs = []
        counter = 10
        with self.graph.as_default():
            epoch_loss = 0
            local_acc = 0
            local_loss = 0
            for step in range(total_step):
                # Get next train batch
                feed_dict = self.feed_dict()
                # Train one batch
                _, _loss, _acc = sess.run([self.optimizer_op, self.loss, self.acc], feed_dict)
                epoch_loss += _loss
                local_loss += _loss
                local_acc += _acc
                # Maybe print log
                if (step + 1) % (batch_per_epoch // counter) == 0:
                    counter += 0
                    # test_step = tf.train.global_step(sess, self.global_step)
                    cur_epoch = step // batch_per_epoch
                    batch = (step % batch_per_epoch) + 1
                    lr = self.learning_rate
                    if isinstance(self.learning_rate, tf.Tensor):
                        lr = sess.run(self.learning_rate, feed_dict)
                    # local_loss = sess.run(self.loss, feed_dict)
                    msg = create_training_log_message(cur_epoch, batch, batch_per_epoch,
                                                      float(local_loss / (batch_per_epoch // counter)),
                                                      local_acc / (batch_per_epoch // counter),
                                                      lr, time.time() - step_time)
                    # print()
                    losses.append(local_loss / (batch_per_epoch // counter))
                    accs.append(local_acc / (batch_per_epoch // counter))
                    local_acc = 0
                    local_loss = 0
                    # if counter % batch_per_epoch == 0:
                    # Do evaluation
                    # loss, acc, acc3 = self.eval(sess, self.test_data_generator, batch_size)
                    # add_evaluation_log_message(msg.eval_message, float(loss), float(acc), float(acc3),
                    #                            time.time() - epoch_time, eval_size)
                    # valid_losses.append(loss)
                    log_beautiful_print(msg)
                    step_time = time.time()

                if (step + 1) % batch_per_epoch == 0:
                    print('{:-^30}'.format('Epoch {:d} Summary'.format(cur_epoch)))
                    print("[Train] Time: {:.1f}s, Avg Loss: {:.4f}".format(time.time() - epoch_time,
                                                                           epoch_loss / batch_per_epoch))
                    epoch_time = time.time()
                    loss, acc, acc3 = self.eval(sess, self.test_data_generator, batch_size)
                    valid_losses.append(loss)
                    print('[Valid] Time: {:.1f}s, Loss: {:.3f}, Acc: {:.2f}%, Acc3: {:.2f}%, eval num: {:d}'.format(
                        time.time() - epoch_time, loss, acc * 100, acc3 * 100, eval_size))
                    print('{:-^30}'.format('Epoch {:d} Done'.format(cur_epoch)))
                    epoch_loss = 0
                    epoch_time = time.time()

                self.on_one_batch(sess, step)
        return losses, valid_losses

    # def on_one_batch(self, sess, step):
    #     """
    #     After each batch, this function will be called, all funcs in self._after_one_batch will be called sequentially.
    #     :param sess: tf session
    #     :param step: current training step
    #     :return: None
    #     """
    #     for func in self._after_one_batch:
    #         func(self, sess, step)
    #
    # def after_one_batch(self, func):
    #     """
    #     Add a hook function func, which will be called after each step
    #     :param func:
    #     :return:
    #     """
    #     self._after_one_batch.append(func)

    def infer_in_batches(self, sess, data, batch_size):
        """
        Get logits and predictions of a dataset by running it in small batches.
        :param sess: the tf.Session() to run the computation
        :param data: data used to infer the logits and predictions
        :param batch_size: size of the batch
        :return: logits and predictions
        """
        size = data.shape[0]

        data_node = self.eval_data_node
        num_label = self.back.output_shape[0]

        predictions = np.ndarray(shape=(size, num_label), dtype=np.float32)
        logits = np.ndarray(shape=(size, num_label), dtype=np.float32)
        for begin in range(0, size, batch_size):
            end = begin + batch_size
            if end > size:
                end = size
            logits[begin:end, :] = sess.run(self.eval_logits, feed_dict={data_node: data[begin:end, ...]})
            predictions[begin:end, :] = sess.run(self.eval_prediction,
                                                 feed_dict={self.eval_logits: logits[begin:end, :]})
        return logits, predictions

    def eval(self, sess, data_generator=None, data=None, labels=None, batch_size=200):
        """
        The evaluating function that will be called at the training API
        :param sess: the tf.Session() to run the computation
        :param data_generator: a data generator
        :param data: data used to evaluate the model
        :param labels: data's corresponding labels used to evaluate the model
        :param batch_size: batch size
        :return: loss accuracy and accuracy-5
        """
        if data_generator is None:
            assert data is not None and labels is not None
            logits, predictions = self.infer_in_batches(sess, data, batch_size)
            loss = sess.run(self.eval_loss, {self.eval_logits: logits, self.eval_labels_node: labels})
            acc, acc3 = sess.run([self.eval_acc, self.eval_acc3],
                                 {self.eval_prediction: predictions, self.eval_labels_node: labels})
            return loss, acc, acc3
        else:
            loss = acc = acc3 = 0
            batch_num = data_generator.n // data_generator.batch_size
            for i in range(batch_num):
                data, label = data_generator.next()
                loss_, acc_, acc3_ = sess.run([self.eval_loss, self.eval_acc, self.eval_acc3],
                                              {self.eval_data_node: data, self.eval_labels_node: label})
                loss += loss_
                acc += acc_
                acc3 += acc3_
            return loss / batch_num, acc / batch_num, acc3 / batch_num

    def infer(self, sess, data_generator=None, data=None, batch_size=200):
        if data_generator is None:
            return self.infer_in_batches(sess, data, batch_size)
        else:
            predictions = []
            logits = []
            # batch_num = math.ceil(data_generator.n / data_generator.batch_size)
            for i in range(0, data_generator.n, data_generator.batch_size):
                data, _ = data_generator.next()
                logit, prediction = sess.run([self.eval_logits, self.eval_prediction],
                                             feed_dict={self.eval_data_node: data})
                predictions.append(prediction)
                logits.append(logit)
            predictions = np.vstack(predictions)
            logits = np.vstack(logits)
            assert_str = 'predictions shape: ' + str(predictions.shape) + 'data_generator.n: ' + str(data_generator.n)

            assert len(predictions) == data_generator.n, assert_str
            return logits, predictions

    @property
    def logdir(self):
        return self._logdir or get_path('./models', self.name_or_scope)

    # @property
    # def train_size(self):
    #     return self.train_data_generator.n // self.train_data_generator.batch_size * self.train_data_generator.batch_size
    #
    # @property
    # def test_size(self):
    #     return self.test_data_generator.n // self.test_data_generator.batch_size * self.test_data_generator.batch_size

    def save(self, path=None):
        """
        Save the trained model to disk
        :param sess: the running Session
        :param path: path
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
        :return: None
        """
        if self.finalized:
            # print("Graph has already been finalized!")
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
            return func(self.sess, *args, **kwargs)
