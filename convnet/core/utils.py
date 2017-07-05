"""
Utilities for convnet.py
"""

import math
import os

import tensorflow as tf

from convnet.utils import cuda_runtime as cuda


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_float('gpu_memory', 1.0,
                          """the fraction of memory that the process is allowed to use in a gpu""")

__str2activation = {
    'linear': lambda x: x,
    'relu': tf.nn.relu,
    'relu6': tf.nn.relu6,
    'elu': tf.nn.elu,
    'tan': tf.tanh,
    'sigmoid': tf.sigmoid
}

__str2loss_func = {
    # Currently not supported
    # 'softmax': tf.nn.softmax_cross_entropy_with_logits,
    # 'sigmoid': tf.nn.sigmoid_cross_entropy_with_logits,
    'sparse_softmax': tf.nn.sparse_softmax_cross_entropy_with_logits
}

__str2learning_rate = {
    'exponential': tf.train.exponential_decay,
    'inverse_time': tf.train.inverse_time_decay,
    'natural_exp': tf.train.natural_exp_decay,
    'piecewise_constant': tf.train.piecewise_constant,
    'polynomial_decay': tf.train.polynomial_decay
}


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


def get_activation(str='relu'):
    """
     A utility function to get specified activation function
    :param str: a string which should be among the keys of __str2activation, or a user defined callable function
    :return: a callable function
    """
    if callable(str):
        return str
    if str in __str2activation:
        return __str2activation[str]
    print('No matching activation function found. Using ReLU by default.\n')
    return __str2activation['relu']


def get_loss_func(str='sparse_softmax'):
    """
    A utility function to get specified loss function
    :param str: a string in the keys of __str2loss_func, or a user defined callable function.
        The function is of type: (logits, labels) -> scalar
    :return: a callable loss function
    """
    if callable(str):
        return str
    if str in __str2loss_func:
        return lambda logits, labels: __str2loss_func[str](labels=labels, logits=logits)
    print('No matching loss function found. Using sparse softmax cross entropy by default.\n')
    return __str2loss_func['sparse_softmax']


def get_learning_rate(update_func, **kwargs):
    if callable(update_func):
        return update_func(kwargs['global_step'])
    if update_func == 'piecewise_constant':
        kwargs['x'] = kwargs.pop('global_step')
        kwargs.pop('decay_steps')
        kwargs.pop('learning_rate')
    if update_func in __str2learning_rate:
        return __str2learning_rate[update_func](**kwargs)
    print('No matching learning rate decay scheme found. Using constant scalar learning rate.\n')
    return kwargs['learning_rate']


def get_optimizer(optimizer='Momentum'):
    """
    Use a string to get specific optimizer used for training
    :param optimizer: a string which should be among the keys of __str2optimizer
    :return: a optimizer instance
    """
    if callable(optimizer):
        return optimizer
    if optimizer in __str2optimizer:
        return __str2optimizer[optimizer]
    print('No matching optimizer found. Using Momentum Optimizer by default.\n')
    return __str2optimizer['Momentum']


def output_shape(input_shape, kernel_shape, strides, padding):
    """
    Given specific conditions calculate the output shape of a convolutional layer
    :param input_shape: input shape, e.g.: [28, 28, 3]
    :param kernel_shape: kernel/filter shape, e.g.: [2,2]
    :param strides: the marching strides shape. e.g.: [1,2,2,1]
    :param padding: a tring indicating padding type. 'SAME' or 'VALID'
    :return: the output shape
    """
    if padding == 'SAME':
        x = math.ceil(input_shape[0] / float(strides[1]))
        y = math.ceil(input_shape[1] / float(strides[2]))
        return x, y

    elif padding == 'VALID':
        x = math.ceil((input_shape[0] - kernel_shape[0] + 1) / float(strides[1]))
        y = math.ceil((input_shape[1] - kernel_shape[1] + 1) / float(strides[2]))
        return x, y


def config_proto():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=cuda._gpu_memory)
    return tf.ConfigProto(device_count={"GPU": 1}, gpu_options=gpu_options, allow_soft_placement=True)
