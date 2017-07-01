import tensorflow as tf

from cnn.convnet.utils import init_tf_environ, get_path
from cnn.data.preprocess import prep_data
from cnn.generate_submission import convnet_submission
from cnn.model import build_model

num_epochs = 10
BATCH_SIZE = 50
EVAL_FREQUENCY = 1

FLAGS = tf.app.flags.FLAGS


def main():
    init_tf_environ(gpu_num=1)
    all_data = prep_data(test=True)
    model = build_model(FLAGS.model, FLAGS.name, *all_data[:2])
    model.restore()
    convnet_submission(model, all_data[2], get_path('submissions/' + model.name_or_scope + '/submission.csv'))
    # model.(BATCH_SIZE, 1, EVAL_FREQUENCY)
    # model.save()

if __name__ == '__main__':
    main()
