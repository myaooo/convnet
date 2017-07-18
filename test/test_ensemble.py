import tensorflow as tf

from convnet.ensemble import EnsembleModel
from convnet.core import ConvNet
from convnet.preprocess import prepare_data_fer2013, IMG_SIZE, CHANNELS
import convnet.model_face as model_pool

def test():
    # model_list = [model_pool.model0, model_pool.model0]
    name_list = ['Test2', 'Test1']
    models = [ConvNet(name) for name in name_list]
    graph = tf.Graph()
    model = EnsembleModel(models, [IMG_SIZE[0], IMG_SIZE[1], CHANNELS], name='ensemble', graph=graph)

    data = prepare_data_fer2013()
    model.fit(data['train'], batch_size=128, checkpoint_per_point=100, verbose_frequency=5)


if __name__ == '__main__':
    test()