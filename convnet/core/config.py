"""
Configurations for convnet
"""

import tensorflow as tf


Float32 = tf.float32
Float64 = tf.float64
# Float = Float32
Int32 = tf.int32
Int64 = tf.int64
# Int = Int32
update_ops = tf.GraphKeys.UPDATE_OPS
global_keys = [tf.GraphKeys.GLOBAL_VARIABLES]
local_keys = [tf.GraphKeys.LOCAL_VARIABLES]
weight_keys = global_keys + [tf.GraphKeys.WEIGHTS]
bias_keys = global_keys + [tf.GraphKeys.BIASES]
l_weight_keys = local_keys + [tf.GraphKeys.WEIGHTS]
l_bias_keys = local_keys + [tf.GraphKeys.BIASES]
save_keys = [tf.GraphKeys.TRAINABLE_VARIABLES, tf.GraphKeys.LOCAL_VARIABLES]
Tensor = tf.Tensor


def data_type():
    return Float32
