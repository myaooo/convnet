"""Trainer class for training a convnet"""

import tensorflow as tf
from cnn.convnet.convnet import ConvNet

from cnn.convnet.preprocess import DataGenerator

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


class Trainer(object):

    def __init__(self, model: ConvNet):
        self.learning_rate = tf.Variable(0., trainable=False)
        self.new_learning_rate = tf.Variable(0., trainable=False)
        self.update_lr_op = tf.assign(self.learning_rate, self.new_learning_rate)
        self.global_step = tf.Variable(0, trainable=False)
        self.epoch = 0
        self._update_lr_func = None
        self.optimizer = None
        self.model = model
        self.train_ops = []
        self.train_logs = []
        self.before_batch_hooks = []
        self.after_batch_hooks = []
        # self.params = params

    def set_optimizer(self, optimizer, *args, **kwargs) -> tf.train.Optimizer:
        self.optimizer = get_optimizer(optimizer)(*args, **kwargs)

    def set_learning_rate(self, learning_rate=0.1, update_func=None):
        if update_func is None:
            self._update_lr_func = lambda: learning_rate
        else:
            self._update_lr_func = eval(update_func)

    def update_lr(self, sess):
        new_lr = self._update_lr_func(self.epoch)
        sess.run(self.update_lr_op, {self.new_learning_rate: new_lr})

    def _prepare_train(self):
        # assert isinstance(self.optimizer, tf.train.Optimizer)
        assert self.model.is_compiled, "model must be compiled before training!"
        optimizer_op = self.optimizer.minimize(self.model.loss, self.global_step)

        self.train_ops.append(optimizer_op)
        self.train_ops.append(self.model.loss)
        self.train_ops.append(self.model.acc)

    def _train_one_epoch(self, sess, train_data_generator):
        for i in range(train_data_generator.epoch_size):
            batch_data, batch_labels = next(train_data_generator)
            feed_dict = {self.model.data_node: batch_data,
                         self.model.labels_node: batch_labels}
            self.before_one_batch()
            logs = sess.run(self.train_ops, feed_dict)
            self.train_logs.append(logs)
            self.after_one_batch()

    def train(self, sess, train_data_generator, num_epochs=20):
        assert isinstance(train_data_generator, DataGenerator)
        self._prepare_train()
        for i in range(num_epochs):
            self.epoch = i
            self._train_one_epoch(sess, train_data_generator)
            self.update_lr(sess)

    def after_one_batch(self, func=None):
        if func is None:
            for f in self.after_batch_hooks:
                f(self)
        else:
            self.after_batch_hooks.append(func)

    def before_one_batch(self, func=None):
        if func is None:
            for f in self.before_batch_hooks:
                f(self)
        else:
            self.before_batch_hooks.append(func)
