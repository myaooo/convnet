"""Trainer class for training a convnet"""

import time
from collections import defaultdict, Counter

import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as tflayers

from convnet.core.convnet import ConvNet
from convnet.core.preprocess import DataGenerator, ImageDataGenerator
from convnet.utils.io_utils import before_save
from convnet.core.config import update_ops

DECAY_STEP = 10000


__str2optimizer = {
    'GradientDescent': tf.train.GradientDescentOptimizer,
    'Adadelta': tf.train.AdadeltaOptimizer,
    'Adagrad': tf.train.AdagradOptimizer,
    'AdagradDA': tf.train.AdagradDAOptimizer,
    'Momentum': tf.train.MomentumOptimizer,
    'Adam': tf.train.AdamOptimizer,
    'Ftrl': tf.train.FtrlOptimizer,
    'ProximalGradientDescent': tf.train.ProximalGradientDescentOptimizer,
    'ProximalAdagrad': tf.train.ProximalAdagradOptimizer,
    'RMSProp': tf.train.RMSPropOptimizer
}


def get_optimizer(optimizer='Momentum'):
    """
    Use a string to get specific optimizer used for training
    :param optimizer: a string which should be among the keys of __str2optimizer
    :return: a optimizer instance
    """
    if isinstance(optimizer, tf.train.Optimizer):
        return optimizer
    if optimizer in __str2optimizer:
        return __str2optimizer[optimizer]
    print('No matching optimizer found. Using Momentum Optimizer by default.\n')
    return __str2optimizer['Momentum']


def get_regularizer(regularizer, scale=0):
    """
    A utility function to get user specified regularizer
    e.g.: get_regularizer('l2', 0.005)
    :param regularizer: of function type (x): (func weights: scalar).
        Take a scalar as input, return a function of type: take a Tensor as input and return a scalar
    :param scale: a scalar
    :return: a function of type: take a list of Tensor as input and return a scalar
    """
    if regularizer is None:
        return lambda x: 0
    if callable(regularizer):
        return regularizer(scale)
    if regularizer == 'l1':
        return tflayers.l1_regularizer(scale)
    if regularizer == 'l2':
        return tflayers.l2_regularizer(scale)
    raise ValueError('regularizer should either be callable or "l1" or "l2"!')
    # _regularizer = _get_regularizer(regularizer, scale)
    # return lambda weight_list: tflayers.apply_regularization(_regularizer, weight_list)


class TrainingRecorder(object):
    """
    A recorder that processes and prints training logs
    """

    def __init__(self, log_keys=None, file_name=None):
        if log_keys is None:
            log_keys = ['loss', 'acc']
        elif isinstance(log_keys, str):
            log_keys = [log_keys]
        self.log_keys = log_keys
        self.training_logs = []
        self.valid_logs = []
        self.log_buffer = []
        self.step = 0
        self.log_per_step = 0
        self.file_name = file_name
        self.verbose_per_step = None
        if file_name is not None:
            before_save(file_name)
        self.timer = None

    def start(self, verbose_per_step):
        self.verbose_per_step = verbose_per_step
        self.timer = time.time()

    def record_step(self, logs):
        self.log_buffer.append(logs)
        self.step += 1
        if len(self.log_buffer) >= self.verbose_per_step:
            record = {}
            for key in self.log_keys:
                record[key] = np.mean([v[key] for v in self.log_buffer])
            record['time'] = time.time() - self.timer
            self.training_logs.append(record)
            self.log_buffer = []
            self.print_record(self.step, record)
            self.timer = time.time()

    def record_validation(self, valid_record):
        self.valid_logs.append(valid_record)
        s = 'Validation'
        for key in self.log_keys:
            s += ' - {:s}: {:.4f}'.format(key, valid_record[key])
        self.output(s)

    def print_record(self, step, record):
        s = 'Step {:d}'.format(step)
        for key in self.log_keys:
            s += ' - {:s}: {:.4f}'.format(key, record[key])
        if 'time' in record:
            s += ' - {:s}: {:.4f}s'.format('time', record['time'])
        self.output(s)

    def output(self, s):
        print(s, flush=True)
        if self.file_name is not None:
            with open(self.file_name, 'a') as f:
                f.write("{:s}\n".format(s))

class Trainer(object):
    def __init__(self, net: ConvNet):
        self.learning_rate = None
        self.new_learning_rate = None
        self.update_lr_op = None
        self.global_step = None
        self._update_lr_func = None
        self.optimizer = None
        self._net = net
        # self.model = net.model
        # assert self.model is not None, 'The model is not trainable! Check if "train" is set to True when compiling!'
        self.train_ops = {}
        self.log_keys = []
        self.regularizers = []
        # self.loss = None
        self.max_epochs = None
        self.need_stop = False
        self._weight_func = None
        self.is_restored = False

    def set_optimizer(self, optimizer, *args, **kwargs):
        self.optimizer = lambda lr: get_optimizer(optimizer)(lr, *args, **kwargs)

    def set_learning_rate(self, learning_rate=0.1, update_func=None):
        if update_func is None:
            self._update_lr_func = lambda step: learning_rate
        else:
            assert callable(update_func), "update_func should be a callable function!"
            self._update_lr_func = update_func

    def weighted_loss(self, weight_func):
        self._weight_func = weight_func

    def add_regularizer(self, regularizer, scale):
        self.regularizers.append(get_regularizer(regularizer, scale))

    def update_lr(self, sess, step):
        new_lr = self._update_lr_func(step)
        sess.run(self.update_lr_op, {self.new_learning_rate: new_lr})
        print("learning rate: {:.4f}".format(new_lr))

    def _prepare_training(self):
        if self.is_restored:
            return
        with self._net.graph.as_default():
            with tf.name_scope(self._net.name + "/training"):
                self.learning_rate = tf.Variable(0., trainable=False, name='learning_rate')
                self.new_learning_rate = tf.Variable(0., trainable=False, name='new_learning_rate')
                self.update_lr_op = tf.assign(self.learning_rate, self.new_learning_rate, name='update_lr_op')
                self.global_step = tf.Variable(0, trainable=False, name='global_step')
                penalty = 0
                if len(self.regularizers):
                    regularizer = tflayers.sum_regularizer(regularizer_list=self.regularizers)
                    penalty = tflayers.apply_regularization(regularizer, tf.get_collection(tf.GraphKeys.WEIGHTS))
                loss = tf.add(self._net.loss, penalty, name='train_loss')

                update_op = tf.get_collection(update_ops, scope=self._net.name_prefix)
                with tf.control_dependencies(update_op):
                    var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self._net.name)
                    self.optimize_op = self.optimizer(self.learning_rate)\
                        .minimize(loss, self.global_step, var_list=var_list, name="train_op")

    def restore_from_graph(self):
        prefix = self._net.name + "/training/"
        with self._net.graph as graph:
            self.learning_rate = graph.get_tensor_by_name(prefix + "learning_rate:0")
            self.new_learning_rate = graph.get_tensor_by_name(prefix + "new_learning_rate:0")
            self.update_lr_op = graph.get_tensor_by_name(prefix + "update_lr_op:0")
            self.global_step = graph.get_tensor_by_name(prefix + "learning_rate:0")
            self.train_ops['optimize'] = graph.get_operation_by_name(prefix + "train_op")

    def _train(self, sess, train_data_generator: DataGenerator, valid_data_generator: DataGenerator,
               max_steps: int, checkpoint_per_step: int, verbose_frequency: int,
               recorder: TrainingRecorder):
        if recorder is None:
            recorder = TrainingRecorder()
        self.train_ops['optimize'] = self.optimize_op
        self.train_ops.update(self._net.get_fetches(recorder.log_keys))
        self.need_stop = False
        self.is_prepared = True
        recorder.start(checkpoint_per_step//verbose_frequency)
        # additional_feed = None
        for step in range(max_steps):
            if self.need_stop:
                return
            batch_data, batch_label = next(train_data_generator)
            if self._weight_func is not None:
                loss_weights = np.array([self._weight_func(self.class_sizes[label]) for label in batch_label])
                loss_weights = loss_weights * self.normalize_factor
            else:
                loss_weights = np.ones((len(batch_label),), dtype=np.float32)
            additional_feed = {self._net.loss_weights: loss_weights, self._net.is_training: True}
            logs = self._net.run(sess, batch_data, batch_label, self.train_ops, additional_feed)
            recorder.record_step(logs)
            if (step + 1) % (checkpoint_per_step) == 0:
                if valid_data_generator is not None:
                    valid_data_generator.reset()
                    valid_record = self._net._eval(sess, valid_data_generator)
                    recorder.record_validation(valid_record)
                self._net.save(global_step=self.global_step)
                self.update_lr(sess, step)
        self._net.save(global_step=self.global_step)

    def train(self, train_data: tuple, valid_data: tuple = None, batch_size: int = 64,
              max_steps: int = 20, checkpoint_per_step: int = 500, verbose_frequency: int = 5,
              recorder: TrainingRecorder = None):
        """
        main api to train the model
        :param train_data: training data. A tuple (X, Y)
        :param valid_data: validation data. A tuple (X, Y)
        :param batch_size: batch size. int
        :param max_steps: the steps to run the training. int
        :param checkpoint_per_step: specify how often to save a checkpoint. int
        :param verbose_frequency: specify how often between checkpoint to verbose. int
        :param recorder: Optional
        :return:
        """

        self._prepare_training()
        train_data_generator = ImageDataGenerator(train_data[0], train_data[1], batch_size, shuffle=True)
        valid_data_generator = None
        if valid_data is not None:
            valid_data_generator = ImageDataGenerator(valid_data[0], valid_data[1], batch_size, epoch_num=1)
        if self._weight_func is not None:
            self.class_sizes = Counter(train_data[1])
            self.class_sizes = np.array([self.class_sizes[c] for c in range(len(self.class_sizes))])
            self.weights = np.array([self._weight_func(c) for c in self.class_sizes])
            self.normalize_factor = np.sum(self.class_sizes) / np.sum(self.class_sizes * self.weights)
        self._net.run_with_context(self._train, train_data_generator, valid_data_generator, max_steps,
                                   checkpoint_per_step, verbose_frequency, recorder)
