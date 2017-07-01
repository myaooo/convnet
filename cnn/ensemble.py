import tensorflow as tf
import numpy as np
from sklearn.linear_model import LogisticRegression

from cnn.data.preprocess import BATCH_SIZE
from cnn.convnet.utils import init_tf_environ, get_path
from cnn.data.preprocess import prep_data
from cnn.generate_submission import generate_submission
from cnn.model import build_model

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('models', '',
                           """A list of model number split by ','""")
tf.app.flags.DEFINE_string('names', '',
                           """A list of names of models""")
tf.app.flags.DEFINE_string('dataset', 'valid',
                           """which set of data to run""")
tf.app.flags.DEFINE_string('out', 'submissions/ensemble.csv',
                           """the path to the output""")
tf.app.flags.DEFINE_boolean('lr', False,
                           """Whether using logistic regression as aggregate learner""")


def ensemble_predict(data_generator, models, weights=None):
    # n_model = len(models)
    predictions = []
    logits = []
    for model in models:
        logit, prediction = model.infer(model.sess, data_generator, batch_size=BATCH_SIZE)
        predictions.append(prediction)
        logits.append(logit)
    if weights is None:
        predictions = np.stack(predictions)
        prediction = np.mean(predictions, axis=0)
    else:
        weights = weights / np.sum(weights)
        for i in range(len(models)):
            predictions[i] = predictions[i] * weights[i]
        predictions = np.stack(predictions)
        prediction = np.sum(predictions, axis=0)
    return prediction


def ensemble_learn(train_data_generator, models):
    # n_model = len(models)
    # learn
    train_predictions = []
    logits = []
    for model in models:
        logit, prediction = model.infer(model.sess, train_data_generator, batch_size=BATCH_SIZE)
        train_predictions.append(prediction[:, 1])
        logits.append(logit)
    train_predictions = np.stack(train_predictions).T
    labels = train_data_generator.y
    solver = LogisticRegression(C=10000, penalty='l2', tol=0.05)
    solver.fit(train_predictions, labels)

    coef = solver.coef_.ravel()
    print("The coeficient of the models are:")
    print(coef)
    return coef


def cal_loss(predictions, labels):
    total_loss = 0
    for i, label in enumerate(labels):
        total_loss -= np.log(predictions[i, label])
    return total_loss/len(labels)


def cal_acc(predictions, labels):
    right_guess = 0.0
    for i, label in enumerate(labels):
        if label == np.argmax(predictions[i, :]):
            right_guess += 1.0
    return right_guess / len(labels)


def ensemble_eval(predictions, labels):

    # predictions = ensemble_predict(data_generator, models, weights=None)
    # labels = data_generator.y
    ensemble_loss = cal_loss(predictions, labels)
    ensemble_acc = cal_acc(predictions, labels)
    print('ensemble loss:', ensemble_loss)
    print('ensemble accuracy:', ensemble_acc)
    return
    # losses = []
    # accs = []
    # for model in models:
    #     loss, acc, _ = model.eval(model.sess, data_generator, batch_size=BATCH_SIZE)
    #     losses.append(loss)
    #     accs.append(acc)


def main():
    init_tf_environ(gpu_num=1)
    all_data = prep_data(test=True, all=FLAGS.train == 'all', shuffle=False)
    models = [int(num) for num in FLAGS.models.split(',')]
    names = FLAGS.names.split(',')
    dataset = FLAGS.dataset
    cnns = []
    weights = None
    for model, name in zip(models, names):
        cnn = build_model(model, name, *all_data[:2])
        cnn.restore()
        cnns.append(cnn)
    if FLAGS.lr:
        weights = ensemble_learn(all_data[0], cnns)
    d = 0 if dataset == 'train' else 1 if dataset == 'valid' else 2
    predictions = ensemble_predict(all_data[d], cnns, weights)
    if dataset == 'test':
        generate_submission(predictions[:, 1], get_path(FLAGS.out))
        return
    ensemble_eval(predictions, all_data[d].y)
    


if __name__ == '__main__':
    main()
