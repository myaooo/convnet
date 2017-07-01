"""Preprocessing helpers for convnet"""

import math


class DataGenerator(object):
    def __init__(self, X, batch_size, epoch_num=math.inf):
        self.X = X
        self.batch_size = batch_size
        self.i = 0
        self.epoch_num = epoch_num
        self._epoch_size = self.n // self.batch_size

    @property
    def n(self):
        return len(self.X)

    @property
    def epoch_size(self):
        return self._epoch_size

    def __next__(self):
        if self.i > self.epoch_num * self.epoch_size:
            raise StopIteration()
        step = self.i % self._epoch_size
        start = step * self.batch_size
        return self.X[start:(start + self.batch_size), :]

    def __iter__(self):
        return self
