"""Trainer class for training a convnet"""

import time
import sys
from collections import defaultdict

import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as tflayers

from convnet.core.convnet import ConvNet, ConvModel
from convnet.core.preprocess import DataGenerator
from convnet.utils.io_utils import before_save
from convnet.core.config import update_ops

DECAY_STEP = 10000

# __str2learning_rate = {
#     'exponential': tf.train.exponential_decay,
#     'inverse_time': tf.train.inverse_time_decay,
#     'natural_exp': tf.train.natural_exp_decay,
#     'piecewise_constant': tf.train.piecewise_constant,
#     'polynomial_decay': tf.train.polynomial_decay
# }
#
#
# def get_learning_rate(update_func, **kwargs):
#     if callable(update_func):
#         return update_func(kwargs['global_step'])
#     if update_func == 'piecewise_constant':
#         kwargs['x'] = kwargs.pop('global_step')
#         kwargs.pop('decay_steps')
#         kwargs.pop('learning_rate')
#     if update_func in __str2learning_rate:
#         return __str2learning_rate[update_func](**kwargs)
#     try:
#         func = eval(update_func)
#         return func
#     except:
#         print('No matching learning rate decay scheme found. Using constant scalar learning rate.\n')
#         return kwargs['learning_rate']


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
        with net.graph.as_default():
            self.learning_rate = tf.Variable(0., trainable=False)
            self.new_learning_rate = tf.Variable(0., trainable=False)
            self.update_lr_op = tf.assign(self.learning_rate, self.new_learning_rate)
            self.global_step = tf.Variable(0, trainable=False)
        self._update_lr_func = None
        self.optimizer = None
        self._net = net
        self.model = net.models['train']
        assert self.model is not None, 'The model is not trainable! Check if "train" is set to True when compiling!'
        self.train_ops = {}
        self.log_keys = []
        self.train_logs = defaultdict(list)
        self.regularizers = []
        self.loss = None
        self.max_epochs = None
        self.need_stop = False

    def set_optimizer(self, optimizer, *args, **kwargs):
        self.optimizer = get_optimizer(optimizer)(self.learning_rate, *args, **kwargs)

    def set_learning_rate(self, learning_rate=0.1, update_func=None):
        if update_func is None:
            self._update_lr_func = lambda step: learning_rate
        else:
            assert callable(update_func), "update_func should be a callable function!"
            self._update_lr_func = update_func

    def add_regularizer(self, regularizer, scale):
        self.regularizers.append(get_regularizer(regularizer, scale))

    def update_lr(self, sess, step):
        new_lr = self._update_lr_func(step)
        sess.run(self.update_lr_op, {self.new_learning_rate: new_lr})

    def _prepare_training(self, log_keys):
        with self._net.graph.as_default():
            regularizer = tflayers.sum_regularizer(regularizer_list=self.regularizers)
            penalty = tflayers.apply_regularization(regularizer, tf.get_collection(tf.GraphKeys.WEIGHTS))
            self.loss = self.model.loss + penalty

            update_op = tf.get_collection(update_ops)
            with tf.control_dependencies(update_op):
                optimizer_op = self.optimizer.minimize(self.loss, self.global_step)

        self.train_ops['optimize'] = optimizer_op
        self.train_ops.update(self.model.get_fetches(log_keys))
        self.need_stop = False

    # def _train_one_step(self, sess, train_data_generator) -> dict:
    #     batch_data, batch_label = next(train_data_generator)
    #     return self.model.run(sess, batch_data, batch_label, self.train_ops)

    def _train(self, sess, train_data_generator: DataGenerator, valid_data_generator: DataGenerator,
               max_steps: int, checkpoint_per_step: int, verbose_frequency: int,
               recorder: TrainingRecorder):
        recorder.start(checkpoint_per_step/verbose_frequency)
        for step in range(max_steps):
            if self.need_stop:
                return
            batch_data, batch_label = next(train_data_generator)
            logs = self.model.run(sess, batch_data, batch_label, self.train_ops)
            recorder.record_step(logs)
            if (step + 1) % (checkpoint_per_step) == 0:
                valid_data_generator.reset()
                if valid_data_generator is not None:
                    valid_record = self._net.models['eval'].eval(sess, valid_data_generator)
                    recorder.record_validation(valid_record)
                self._net.save()
                self.update_lr(sess, step)

    def train(self, train_data_generator: DataGenerator, valid_data_generator: DataGenerator = None,
              max_steps: int = 20, checkpoint_per_step: int = 500, verbose_frequency: int = 5,
              recorder: TrainingRecorder = None):
        if recorder is None:
            recorder = TrainingRecorder()
        self._prepare_training(recorder.log_keys)
        self._net.run_with_context(self._train, train_data_generator, valid_data_generator, max_steps,
                                   checkpoint_per_step, verbose_frequency, recorder)
