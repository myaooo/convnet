"""
Base class of classifier
"""

import tensorflow as tf


class Classifier(object):

    def top_k_acc(self, predictions, labels, k=1, name=None):
        in_k = tf.nn.in_top_k(predictions, labels, k)
        return tf.reduce_mean(tf.to_float(in_k), name=name)

    def in_top_k(self, predictions, labels, k=1, name=None):
        return tf.nn.in_top_k(predictions,labels, k, name)
