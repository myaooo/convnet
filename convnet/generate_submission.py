from convnet.utils import lists2csv
from convnet.core.convnet import ConvNet
from convnet.data.preprocess import BATCH_SIZE


def generate_submission(predictions, file_name='submission.csv'):
    ids = list(range(1, len(predictions)+1))
    data = list(zip(ids, predictions))
    lists2csv(data, file_name, ['id', 'label'])
    print("submission saved to {:s}".format(file_name))


def convnet_submission(model, test_data, file_name='submission.csv'):
    assert isinstance(model, ConvNet), "model should be an instance of ConvNet"
    _, predictions = model.infer(model.sess, test_data, batch_size=BATCH_SIZE)
    predictions = predictions[:, 1]
    generate_submission(predictions, file_name)
