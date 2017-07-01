"""
A Recoder Class, which is used to interpolate in a ConvNet for recording training and evaluation information
"""

import tensorflow as tf
from . import convnet as cnet


class ConvRecorder:
    """
    A Recorder of ConvNet
    """
    def __init__(self, convnet, record_dir, record_per_steps=100):
        assert isinstance(convnet, cnet.ConvNet)
        self.convnet = convnet
        self.record_dir = record_dir
        self.record_per_steps = record_per_steps
        self.is_init = False
        self.summary_list = []
        self.summary = None
        self.convnet.after_one_batch(self)
        self.record_dir = record_dir
        self.writer = None

    def __call__(self, convnet, sess, step):
        """
        This function will be installed into ConvNet hooks
        :param convnet: the convnet uses this recorder
        :param sess: the tf.Session that runs the computation
        :param step: gloabal_step
        :return:
        """
        assert convnet == self.convnet
        if not self.is_init:
            self._record(self.convnet)
            self.is_init = True
        if step % self.record_per_steps == 0:
            summary = sess.run(self.summary, convnet.feed_dict())
            self.writer.add_summary(summary, step)
            self.writer.flush()

    def _record(self, convnet):
        """
        Do record initializations on the convnet
        :param convnet: the convnet to be recorded
        :return:
        """
        assert convnet.is_compiled

        self.writer = tf.summary.FileWriter(self.record_dir, convnet.sess.graph)
        layer = convnet.front
        while layer != convnet.back:
            if isinstance(layer, cnet.ConvLayer):
                with tf.variable_scope(layer.scope_name):
                    self.add_tensor_summary('filters', layer.filters)
                    if layer.bias is not None:
                        self.add_tensor_summary('bias', layer.bias)
            elif isinstance(layer, cnet.FullyConnectedLayer):
                with tf.variable_scope(layer.scope_name):
                    self.add_tensor_summary('filters', layer.weights)
                    if layer.bias is not None:
                        self.add_tensor_summary('bias', layer.bias)
            layer = layer.next
        self.summary = tf.summary.merge(self.summary_list)
        convnet.after_one_batch(self)

    def add_tensor_summary(self, name, tensor, description=None):
        """
        Add a tensor to recording list
        :param name:
        :param tensor:
        :param description:
        :return:
        """
        summary = tf.summary.tensor_summary(name, tensor, description)
        self.summary_list.append(summary)

