"""Preprocessing helpers for convnet"""

import math
from collections import Counter

import numpy as np


class DataGenerator(object):
    def __init__(self, X, batch_size, epoch_num=math.inf, shuffle=False):
        self.X = X
        self.batch_size = batch_size
        self.i = 0
        self.epoch_num = epoch_num
        self._epoch_size = self.n // self.batch_size
        self.indices = np.arange(0, self.n)
        self.shuffle = shuffle
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

    def get_indices_of_next_batch(self):
        """
        A method to compute the indices of the next batch.
        This method is intended to be called by __next__()
        :return:
        """
        if self.i >= self.epoch_num * self.epoch_size:
            raise StopIteration()
        step = self.i % self._epoch_size
        start = step * self.batch_size
        indices = self.indices[start:(start + self.batch_size)]
        if (self.i+1) % self.epoch_size == 0:
            indices = self.indices[start:]
            if self.shuffle: # shuffle again
                np.random.shuffle(self.indices)
        return indices

    def __next__(self):
        indices = self.get_indices_of_next_batch()
        self.i += 1
        return self.X[indices]

    def __iter__(self):
        return self

    def reset(self):
        self.i = 0


class ImageDataGenerator(DataGenerator):
    def __init__(self, X, Y, batch_size, epoch_num=math.inf, shuffle=False):
        super().__init__(X, batch_size, epoch_num=epoch_num, shuffle=shuffle)
        self.Y = Y

    def __next__(self):
        indices = self.get_indices_of_next_batch()
        self.i += 1
        return self.X[indices], self.Y[indices]


class RandomImageDataGenerator(ImageDataGenerator):
    def __init__(self, X, Y, batch_size, epoch_num=math.inf, weight_func=None):
        super().__init__(X, Y, batch_size, epoch_num=epoch_num, shuffle=False)
        if weight_func is None:
            weight_func = lambda size: 1/math.sqrt(size)
        class_sizes = Counter(Y)
        # classes = sorted(class_sizes)
        # class_sizes = [class_sizes[c] for c in classes]
        p = np.array([weight_func(class_sizes[y]) for y in Y])
        self.p = p / p.sum()

    def get_indices_of_next_batch(self):
        return np.random.choice(self.indices, self.batch_size, p=self.p)
