"""Preprocessing helpers for convnet"""

import math

import numpy as np


class DataGenerator(object):
    def __init__(self, X, batch_size, epoch_num=math.inf, shuffle=False):
        self.X = X
        self.batch_size = batch_size
        self.i = 0
        self.epoch_num = epoch_num
        self._epoch_size = self.n // self.batch_size
        self.indices = np.arange(0, self.n)
        if shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        return self._epoch_size

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
        return self.X[self.indices[start:(start + self.batch_size)]]

    def __iter__(self):
        return self

    def reset(self):
        self.i = 0


class ImageDataGenerator(DataGenerator):
    def __init__(self, X, Y, batch_size, epoch_num=math.inf, shuffle=False):
        super().__init__(X, batch_size, epoch_num=epoch_num, shuffle=shuffle)
        self.Y = Y

    def __len__(self):
        return self._epoch_size

    def __next__(self):
        if self.i > self.epoch_num * self.epoch_size:
            raise StopIteration()
        step = self.i % self._epoch_size
        start = step * self.batch_size
        indices = self.indices[start:(start + self.batch_size)]
        return self.X[indices], self.Y[indices]

    def __iter__(self):
        return self
