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

__str2loss_func = {
    # Currently not supported
    # 'softmax': tf.nn.softmax_cross_entropy_with_logits,
    # 'sigmoid': tf.nn.sigmoid_cross_entropy_with_logits,
    'sparse_softmax': tf.losses.sparse_softmax_cross_entropy
}


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
        return lambda logits, labels, weights=1.0: __str2loss_func[str](labels=labels, logits=logits, weights=weights)
    print('No matching loss function found. Using sparse softmax cross entropy by default.\n')
    return __str2loss_func['sparse_softmax']


def config_proto():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=cuda._gpu_memory)
    return tf.ConfigProto(device_count={"GPU": 1}, gpu_options=gpu_options, allow_soft_placement=True)
