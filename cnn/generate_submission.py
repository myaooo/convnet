import io
import csv

from cnn.convnet.utils import before_save
from cnn.convnet.convnet import ConvNet
from cnn.data.preprocess import BATCH_SIZE


def lists2csv(lists, file_path, header=None, delimiter=',', encoding=None):
    with io.StringIO() as s_io:
        writer = csv.writer(s_io, delimiter=delimiter)
        if header is not None:
            writer.writerow(header)
        for ls in lists:
            writer.writerow([str(i) for i in ls])
        write2file(s_io, file_path, 'w', encoding=encoding)


def write2file(s_io, file_path, mode, encoding=None):
    """
    This is a wrapper function for writing files to disks,
    it will automatically check for dir existence and create dir or file if needed
    :param s_io: a io.StringIO instance or a str
    :param file_path: the path of the file to write to
    :param mode: the writing mode to use
    :return: None
    """
    before_save(file_path)
    with open(file_path, mode, encoding=encoding) as f:
        if isinstance(s_io, io.StringIO):
            f.write(s_io.getvalue())
        else:
            f.write(s_io)


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
