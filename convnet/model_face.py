import tensorflow as tf

from convnet.core import ConvNet, Trainer
from convnet.utils import init_tf_environ
from convnet.preprocess import IMG_SIZE, CHANNELS, NUM_LABELS, prepare_data_fer2013, TRAIN_SIZE

# from convnet.generate_submission import lists2csv

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('model', 'model1',
                            """The number of model as defined in the script""")
tf.app.flags.DEFINE_integer('epoch', 30,
                            """The number of epochs to run""")

tf.app.flags.DEFINE_string('name', '',
                           """The name of the model""")

tf.app.flags.DEFINE_boolean('train', True,
                            """Set this flag to train""")

tf.app.flags.DEFINE_boolean('test', True,
                            """Set this flag to run the test""")

tf.app.flags.DEFINE_boolean('weighted_loss', False,
                            """Whether to use weighted loss""")

tf.app.flags.DEFINE_integer('checkpoint_per_step', 500,
                            """The interval for saving and validating model""")

tf.app.flags.DEFINE_string('lr_protocol', 'medium',
                           """The protocol of learning rate during training""")

tf.app.flags.DEFINE_integer('batch_size', 128,
                            """batch size""")

# num_epochs = 45
# EVAL_FREQUENCY = 1
N = TRAIN_SIZE


def build_model(model_no, name):
    print("Building model", FLAGS.model)
    # models = [model0, model1, model2, model3, model4]
    # model = models[model_no](name)
    model = eval(FLAGS.model + '("' + FLAGS.name + '")')
    model.loss_func = 'sparse_softmax'
    model.compile()
    return model


def model0(name=''):
    # Test
    model = ConvNet(name or 'Test')
    model.push_input_layer(dshape=[IMG_SIZE[0], IMG_SIZE[1], CHANNELS])
    model.push_augment_layer(4, 4, True, True)
    model.push_conv_layer(filter_size=[3, 3], out_channels=16, strides=[1, 1], activation='linear', has_bias=False)
    model.push_batch_norm_layer(activation='relu')
    model.push_pool_layer('max', [2, 2], strides=[2, 2])
    model.push_conv_layer(filter_size=[3, 3], out_channels=16, strides=[1, 1], activation='linear')
    model.push_batch_norm_layer(activation='relu')
    model.push_pool_layer('max', kernel_size=[2, 2], strides=[2, 2])
    model.push_conv_layer([3, 3], out_channels=32, strides=[1, 1], activation='linear')
    model.push_batch_norm_layer(activation='relu')
    model.push_pool_layer('avg', kernel_size=[int(IMG_SIZE[0] / 4), int(IMG_SIZE[1] / 4)],
                          strides=[int(IMG_SIZE[0] / 4), int(IMG_SIZE[1] / 4)])
    model.push_flatten_layer()
    model.push_fully_connected_layer(out_channels=NUM_LABELS, activation='linear')

    return model


def model1(name=''):
    # Network in Network
    model = ConvNet(name or 'NIN-test')
    model.push_input_layer(dshape=[IMG_SIZE[0], IMG_SIZE[1], CHANNELS])
    model.push_augment_layer(4, 4, True, True)
    model.push_conv_layer(filter_size=[3, 3], out_channels=32, strides=[1, 1], activation='linear', has_bias=False)
    model.push_batch_norm_layer(activation='relu')
    model.push_conv_layer(filter_size=[3, 3], out_channels=32, strides=[1, 1], activation='linear', has_bias=False)
    model.push_batch_norm_layer(activation='relu')
    model.push_pool_layer('max', [2, 2], strides=[2, 2])
    model.push_conv_layer(filter_size=[3, 3], out_channels=64, strides=[1, 1], activation='linear')
    model.push_batch_norm_layer(activation='relu')
    model.push_conv_layer(filter_size=[1, 1], out_channels=64, strides=[1, 1], activation='linear')
    model.push_batch_norm_layer(activation='relu')
    model.push_conv_layer(filter_size=[1, 1], out_channels=64, strides=[1, 1], activation='linear')
    model.push_batch_norm_layer(activation='relu')
    model.push_pool_layer('max', kernel_size=[2, 2], strides=[2, 2])
    model.push_conv_layer([3, 3], out_channels=128, strides=[1, 1], activation='linear')
    model.push_batch_norm_layer(activation='relu')
    model.push_conv_layer(filter_size=[1, 1], out_channels=128, strides=[1, 1], activation='linear')
    model.push_batch_norm_layer(activation='relu')
    model.push_conv_layer(filter_size=[1, 1], out_channels=128, strides=[1, 1], activation='linear')
    model.push_batch_norm_layer(activation='relu')
    model.push_pool_layer('avg', kernel_size=[int(IMG_SIZE[0] / 4), int(IMG_SIZE[1] / 4)],
                          strides=[int(IMG_SIZE[0] / 4), int(IMG_SIZE[1] / 4)])
    model.push_flatten_layer()
    model.push_fully_connected_layer(out_channels=NUM_LABELS, activation='linear')

    return model


def model2(name=''):
    # Network in Network
    model = ConvNet(name or 'NIN2')
    model.push_input_layer(dshape=[IMG_SIZE[0], IMG_SIZE[1], CHANNELS])
    model.push_augment_layer(4, 4, True, True)
    model.push_conv_layer(filter_size=[3, 3], out_channels=16, strides=[2, 2], activation='linear', has_bias=False)
    model.push_batch_norm_layer(activation='relu')
    model.push_conv_layer(filter_size=[3, 3], out_channels=32, strides=[1, 1], activation='linear', has_bias=False)
    model.push_batch_norm_layer(activation='relu')
    model.push_conv_layer(filter_size=[3, 3], out_channels=32, strides=[1, 1], activation='linear', has_bias=False)
    model.push_batch_norm_layer(activation='relu')
    model.push_pool_layer('max', [2, 2], strides=[2, 2])
    model.push_conv_layer(filter_size=[3, 3], out_channels=64, strides=[1, 1], activation='linear')
    model.push_batch_norm_layer(activation='relu')
    model.push_conv_layer(filter_size=[1, 1], out_channels=64, strides=[1, 1], activation='linear')
    model.push_batch_norm_layer(activation='relu')
    model.push_conv_layer(filter_size=[1, 1], out_channels=64, strides=[1, 1], activation='linear')
    model.push_batch_norm_layer(activation='relu')
    model.push_pool_layer('max', kernel_size=[2, 2], strides=[2, 2])
    model.push_conv_layer([3, 3], out_channels=128, strides=[1, 1], activation='linear')
    model.push_batch_norm_layer(activation='relu')
    model.push_conv_layer(filter_size=[1, 1], out_channels=128, strides=[1, 1], activation='linear')
    model.push_batch_norm_layer(activation='relu')
    model.push_conv_layer(filter_size=[1, 1], out_channels=128, strides=[1, 1], activation='linear')
    model.push_batch_norm_layer(activation='relu')
    model.push_pool_layer('avg', kernel_size=[int(IMG_SIZE[0] / 8), int(IMG_SIZE[1] / 8)],
                          strides=[int(IMG_SIZE[0] / 8), int(IMG_SIZE[1] / 8)])
    model.push_flatten_layer()
    model.push_fully_connected_layer(out_channels=NUM_LABELS, activation='linear')

    return model


def model3(name=''):
    # test resnet
    model = ConvNet(name or 'ResNet')
    model.push_input_layer(dshape=[IMG_SIZE[0], IMG_SIZE[1], CHANNELS])
    model.push_augment_layer(4, 4, True, True)
    # model.push_conv_layer(filter_size=[7, 7], out_channels=64, strides=[1, 1], activation='linear', has_bias=False)
    # model.push_conv_layer(filter_size=[3, 3], out_channels=16, strides=[1, 1], activation='linear', has_bias=False)
    model.push_batch_norm_layer(activation='relu')
    # model.push_pool_layer('max', [2, 2], strides=[2, 2])
    model.push_res_layer([3, 3], 32, strides=[1, 1], activation='relu', activate_before_residual=False)
    for i in range(2):
        model.push_res_layer([3, 3], 32, strides=[1, 1], activation='relu')
    model.push_batch_norm_layer(activation='relu')
    model.push_res_layer([3, 3], 64, strides=[2, 2], activation='relu', activate_before_residual=False)
    for i in range(3):
        model.push_res_layer([3, 3], 64, strides=[1, 1], activation='relu')
    model.push_batch_norm_layer(activation='relu')
    model.push_res_layer([3, 3], 128, strides=[2, 2], activation='relu', activate_before_residual=False)
    for i in range(5):
        model.push_res_layer([3, 3], 128, strides=[1, 1], activation='relu')
    model.push_batch_norm_layer(activation='relu')
    model.push_res_layer([3, 3], 256, strides=[2, 2], activation='relu', activate_before_residual=False)
    for i in range(2):
        model.push_res_layer([3, 3], 256, strides=[1, 1], activation='relu')
    model.push_batch_norm_layer(activation='relu')
    model.push_pool_layer('avg', kernel_size=[int(IMG_SIZE[0] / 8), int(IMG_SIZE[1] / 8)],
                          strides=[int(IMG_SIZE[0] / 8), int(IMG_SIZE[1] / 8)])
    model.push_dropout_layer(0.8)
    model.push_flatten_layer()
    model.push_fully_connected_layer(NUM_LABELS, activation='linear', has_bias=True)

    return model


def model4(name=''):
    # test resnet
    model = ConvNet(name or 'ResBN')
    model.push_input_layer(dshape=[IMG_SIZE[0], IMG_SIZE[1], CHANNELS])
    model.push_augment_layer(4, 4, True, True)
    # model.push_conv_layer(filter_size=[7, 7], out_channels=64, strides=[1, 1], activation='linear', has_bias=False)
    model.push_conv_layer(filter_size=[3, 3], out_channels=16, strides=[1, 1], activation='linear', has_bias=False)
    model.push_batch_norm_layer(activation='relu')
    # model.push_pool_layer('max', [2, 2], strides=[2, 2])
    model.push_res_bn_layer([3, 3], 64, strides=[1, 1], activation='relu', activate_before_residual=False)
    for i in range(2):
        model.push_res_bn_layer([3, 3], 64, strides=[1, 1], activation='relu')
    model.push_batch_norm_layer(activation='relu')
    model.push_res_bn_layer([3, 3], 128, strides=[2, 2], activation='relu', activate_before_residual=False)
    for i in range(3):
        model.push_res_bn_layer([3, 3], 128, strides=[1, 1], activation='relu')
    model.push_batch_norm_layer(activation='relu')
    model.push_res_bn_layer([3, 3], 256, strides=[2, 2], activation='relu', activate_before_residual=False)
    for i in range(10):
        model.push_res_bn_layer([3, 3], 256, strides=[1, 1], activation='relu')
    model.push_batch_norm_layer(activation='relu')
    model.push_res_bn_layer([3, 3], 512, strides=[2, 2], activation='relu', activate_before_residual=False)
    for i in range(3):
        model.push_res_bn_layer([3, 3], 512, strides=[1, 1], activation='relu')
    model.push_batch_norm_layer(activation='relu')
    model.push_pool_layer('avg', kernel_size=[int(IMG_SIZE[0] / 8), int(IMG_SIZE[1] / 8)],
                          strides=[int(IMG_SIZE[0] / 8), int(IMG_SIZE[1] / 8)])
    model.push_flatten_layer()
    model.push_fully_connected_layer(NUM_LABELS, activation='linear', has_bias=True)

    return model


def get_lr_protocol(protocol, epoch_size):
    if protocol == 'small':
        return lambda step: \
            0.1 if step < 10 * epoch_size else \
            0.01 if step < 15 * epoch_size else \
            0.001 if step < 20 * epoch_size else 0.0001
    if protocol == 'medium':
        return lambda step: \
            0.1 if step < 20 * epoch_size else \
            0.01 if step < 30 * epoch_size else \
            0.001 if step < 40 * epoch_size else 0.0001
    if protocol == 'large':
        return lambda step: \
            0.1 if step < 30 * epoch_size else \
            0.01 if step < 45 * epoch_size else \
            0.001 if step < 60 * epoch_size else 0.0001
    else:
        raise ValueError("argument 'protocol' needs to be 'small' or 'medium' or 'large'!")


def evaluate(model, data, batch_size):
    logs = model.eval(data, batch_size)
    print('[Test Set] Loss: {:.3f}, Acc: {:.2%}, eval num: {:d}'.format(
        logs['loss'], logs['acc'], len(data[0]) // batch_size * batch_size ))


def train(model, train_data, valid_data, batch_size, epoch, protocol):
    print("prepare training....")
    epoch_size = N // batch_size
    trainer = Trainer(model)
    trainer.add_regularizer('l2', 1e-3)
    trainer.set_learning_rate(update_func=get_lr_protocol(protocol, epoch_size))
    trainer.set_optimizer('Momentum', 0.9)
    if FLAGS.weighted_loss:
        trainer.weighted_loss(lambda size: 1/(size))
    print("start training....")
    trainer.train(train_data, valid_data, batch_size,
                  epoch * N // batch_size, FLAGS.checkpoint_per_step, 5)


def main():
    init_tf_environ(gpu_num=1)
    all_data = prepare_data_fer2013(train=FLAGS.train, valid=FLAGS.train, test=FLAGS.test)
    model = build_model(FLAGS.model, FLAGS.name)
    if FLAGS.train:
        train(model, all_data['train'], all_data['valid'], FLAGS.batch_size, FLAGS.epoch, FLAGS.lr_protocol)
        evaluate(model, all_data['test'], FLAGS.batch_size)
    if FLAGS.test:
        model2 = ConvNet(FLAGS.name)
        model2.compile_from_meta()
        model2.restore_weights()
        evaluate(model2, all_data['test'], FLAGS.batch_size)



if __name__ == '__main__':
    main()
