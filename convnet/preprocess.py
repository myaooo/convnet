import math
from collections import Counter

import numpy as np
# from keras.preprocessing.image import ImageDataGenerator

from convnet.core.preprocess import ImageDataGenerator, RandomImageDataGenerator
from convnet.utils import get_path, maybe_calculate

IMG_SIZE = (48, 48)
# BATCH_SIZE = 64
CHANNELS = 1
NUM_LABELS = 7

TRAIN_SIZE = 28709
TEST_SIZE = VALID_SIZE = 3589


SEED = None


def preprocess_fer2013(file):
    print("Preprocessing...")
    with open(file, 'r') as f:
        raw_data_str = f.read()
    rows = raw_data_str.split('\n')[1:]
    raw_data = [row.split(',') for row in rows if row != '']
    assert len(raw_data) == 35887
    counter = Counter([row[2] for row in raw_data])
    print(counter)
    dataset = {}
    for key, tag in zip(['train', 'valid', 'test'], ['Training', 'PublicTest', 'PrivateTest']):
        _data = [row[:2] for row in raw_data if row[2] == tag]
        labels, data = zip(*_data)
        data = [[int(d) for d in im.split(' ')] for im in data]
        data = np.array(data, dtype=np.uint8).reshape((-1, IMG_SIZE[0], IMG_SIZE[1], 1))
        print(data.shape)
        # data = np.expand_dims(data, axis=2)
        labels = np.array(labels, dtype=np.int32)
        dataset[key] = {'data': data, 'labels': labels}
    # train = [row[:2] for row in raw_data if row[2] == 'Training']
    # valid = [row[:2] for row in raw_data if row[2] == 'PublicTest']
    # test = [row[:2] for row in raw_data if row[2] == 'PrivateTest']
    return dataset


def maybe_preprocess_fer2013():
    tmp_file = get_path('data/tmp', 'dataset.pkl')
    dataset = maybe_calculate(tmp_file, preprocess_fer2013, get_path('data/fer2013', 'fer2013.csv'))
    return dataset


# def generate_data(X, y, batch_size=BATCH_SIZE, train=True):
#     """
#     Using the returned data and label from maybe_preprocess / format_data,
#     return a keras data generator.
#     Only intended to use for training data
#     :param X: a 4D array, formatted data
#     :param y: a 1D array, label array
#     :param batch_size:
#     :param train
#     :return: a keras generator
#     """
#     if train:
#         data_generator = RandomImageDataGenerator(
#             X, y, epoch_num=math.inf, batch_size=batch_size, weight_func=lambda size: 1/math.sqrt(size))
#     else:
#         data_generator = ImageDataGenerator(
#             X, y, epoch_num=1, batch_size=batch_size, shuffle=False)
#     return data_generator


def prepare_data_fer2013(train=True, valid=True, test=False):
    dataset = maybe_preprocess_fer2013()
    data_flows = {}
    for set_tag, need_set in zip(['train', 'valid', 'test'], [train, valid, test]):
        if not need_set:
            continue
        data_labels = dataset[set_tag]
        data = data_labels['data'].astype(np.float32) / 255 - 0.5
        labels = data_labels['labels']
        # data_flows[set_tag] = generate_data(data, labels, batch_size=batch_size, train=set_tag=='train')
        data_flows[set_tag] = (data, labels)
    return data_flows


if __name__ == "__main__":
    dataset = maybe_preprocess_fer2013()
    print(dataset.keys())
