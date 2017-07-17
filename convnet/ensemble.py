import collections
import tensorflow as tf
import numpy as np
from sklearn.linear_model import LogisticRegression

from convnet.core import ConvNet, Trainer, data_type, top_k_acc
from convnet.core.config import Int32, save_keys
from convnet.utils import init_tf_environ, get_path
from convnet.preprocess import prepare_data_fer2013
from convnet.model_face import build_model

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


class EnsembleModel(ConvNet):
    def __init__(self, nets, input_shape, name='ensemble', dtype=data_type(), graph=None, logdir=None):
        # check model legality
        assert isinstance(nets, collections.Iterable)
        nets = list(nets)
        super().__init__(name, dtype, graph, logdir)
        self.nets = nets
        # Build ensemble graph
        self.push_input_layer(input_shape)
        self.loss_func = 'sparse_softmax'
        self.compile()
        self.trainer = Trainer(self)
        self.trainer.set_optimizer('Momentum', 0.8)
        # self.trainer.set_learning_rate()

    def __call__(self, data, train):
        results = []
        for net in self.nets:
            result = net.prediction
            results.append(result)
        # results = [net(data, train) for net in self.nets]
        output_shape = results[0].shape.as_list()[-1]
        results = tf.stack(results, len(results[0].shape))
        results = tf.reshape(results, [-1, len(self.nets)])
        results = tf.nn.xw_plus_b(results, self.weights, self.bias)
        results = tf.reshape(results, [-1, output_shape])
        return results

    def compile(self):

        with self.graph.as_default():
            with tf.variable_scope(self.name):
                self.is_training = tf.placeholder(tf.bool, shape=[], name='is_training')
                self.weights = \
                    tf.get_variable('weights', shape=[len(self.nets), 1], dtype=self.dtype,
                                    initializer=tf.uniform_unit_scaling_initializer(factor=1.0, dtype=self.dtype))
                self.bias = tf.get_variable('bias', shape=[1], dtype=self.dtype,
                                            initializer=tf.constant_initializer(0.01, dtype=self.dtype))
                with tf.name_scope("model"):
                    self.data_node = tf.placeholder(data_type(), [None] + self.front.output_shape, name='data')
                    self.label_node = tf.placeholder(Int32, [None, ], name='label')
        for net in self.nets:
            net.graph = self.graph
            net.compile_from_meta(input_map={net.name_prefix + "data:0": self.data_node,
                                             net.name_prefix + "label:0": self.label_node,
                                             net.name + "/is_training:0": self.is_training})
        with self.graph.as_default():
            with tf.variable_scope(self.name):
                with tf.name_scope("model"):
                    # self.data_node = tf.placeholder(data_type(), [None] + self.front.output_shape, name='data')
                    # self.label_node = tf.placeholder(Int32, [None, ], name='label')
                    logits = self(self.data_node, self.is_training)
                    # logits = self(self.data_node, train=self.is_training)
                    self.prediction = tf.nn.softmax(logits, name='prediction')
                    self.acc = top_k_acc(self.prediction, self.label_node, 1, name='acc')
                    self.acc3 = top_k_acc(self.prediction, self.label_node, 3, name='acc3')
                    self.loss_weights = tf.placeholder(dtype=data_type(), shape=[None], name='loss_weights')
                    self.loss = tf.reduce_mean(self.loss_func(logits, self.label_node, self.loss_weights),
                                               name='loss')
                # Compile
                self.is_compiled = True

    @property
    def sess(self):
        if self._sess is None or self._sess._closed:
            with self.graph.as_default():
                variables = []
                for key in save_keys:
                    variables += tf.get_collection(key, self.name)
                self._saver = tf.train.Saver(variables)
                var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
                init_op = tf.variables_initializer(var_list=var_list)
                self._sess = tf.Session(graph=self.graph)
                self._sess.run(init_op)
                for net in self.nets:
                    net.restore_weights(self._sess)
        return self._sess

    def fit(self, data, batch_size, max_steps=10000, lr=1.0, decay=0.9, checkpoint_per_point=500,
            verbose_frequency=5):
        train_size = len(data[0])
        epoch_size = train_size // batch_size
        self.trainer.set_learning_rate(update_func=lambda step: lr * decay ** (step // epoch_size))
        # predictions = []
        # # data_generator = DataGenerator(data, batch_size, 1)
        # for net in self.nets:
        #     prediction = net._infer(self.sess, data=data[0], batch_size=batch_size)
        #     predictions.append(prediction)
        # predictions = np.stack(predictions, 2)
        # labels = data[1]
        # print('prediction shape:', predictions.shape)
        self.trainer.train(data, batch_size=batch_size, max_steps=max_steps,
                           checkpoint_per_step=checkpoint_per_point, verbose_frequency=verbose_frequency)
        self.save(self.sess)


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
    return total_loss / len(labels)


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
    all_data = prepare_data_fer2013(test=True)
    models = [int(num) for num in FLAGS.models.split(',')]
    names = FLAGS.names.split(',')
    dataset = FLAGS.dataset
    cnns = []
    weights = None
    for model, name in zip(models, names):
        cnn = build_model(model, name, *all_data[:2])
        cnn.restore_weights()
        cnns.append(cnn)
    if FLAGS.lr:
        weights = ensemble_learn(all_data[0], cnns)
    d = 0 if dataset == 'train' else 1 if dataset == 'valid' else 2
    predictions = ensemble_predict(all_data[d], cnns, weights)
    ensemble_eval(predictions, all_data[d].y)


if __name__ == '__main__':
    main()
