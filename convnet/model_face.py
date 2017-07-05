import tensorflow as tf

from convnet.core import ConvNet, Trainer
from convnet.core.recorder import ConvRecorder
from convnet.utils import init_tf_environ, get_path, before_save, lists2csv
from convnet.preprocess import IMG_SIZE, CHANNELS, NUM_LABELS, prepare_data_fer2013, BATCH_SIZE, TRAIN_SIZE
# from convnet.generate_submission import lists2csv

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('model', 1,
                            """The number of model as defined in the script""")
tf.app.flags.DEFINE_integer('epoch', 30,
                            """The number of epochs to run""")

tf.app.flags.DEFINE_string('name', '',
                           """The name of the model""")

tf.app.flags.DEFINE_string('task', 'train,test',
                           """set to "test" if you only want to test""")

# num_epochs = 45
# EVAL_FREQUENCY = 1
N = TRAIN_SIZE


def build_model(model_no, name):
    print("Building model", FLAGS.model)
    models = [model0, model1, model2, model3, model4]
    model = models[model_no](name)
    model.compile()
    return model


def model0(name=''):
    # Test
    model = ConvNet(name or 'Test')
    model.push_input_layer(dshape=[None, IMG_SIZE[0], IMG_SIZE[1], CHANNELS])
    model.push_augment_layer(4, 4, True, True)
    model.push_conv_layer(filter_size=[3, 3], out_channels=32, strides=[1, 1], activation='linear', has_bias=False)
    model.push_batch_norm_layer(activation='relu')
    model.push_pool_layer('max', [2, 2], strides=[2, 2])
    model.push_conv_layer(filter_size=[3, 3], out_channels=32, strides=[1, 1], activation='linear')
    model.push_batch_norm_layer(activation='relu')
    model.push_pool_layer('max', kernel_size=[2, 2], strides=[2, 2])
    model.push_conv_layer([3, 3], out_channels=64, strides=[1, 1], activation='linear')
    model.push_batch_norm_layer(activation='relu')
    model.push_pool_layer('avg', kernel_size=[int(IMG_SIZE[0] / 4), int(IMG_SIZE[1] / 4)],
                          strides=[int(IMG_SIZE[0] / 4), int(IMG_SIZE[1] / 4)])
    model.push_flatten_layer()
    model.push_fully_connected_layer(out_channels=NUM_LABELS, activation='linear')

    model.loss_func = 'sparse_softmax'

    return model


def model1(name=''):
    # Network in Network
    model = ConvNet(name or 'NIN-test')
    model.push_input_layer(dshape=[None, IMG_SIZE[0], IMG_SIZE[1], CHANNELS])
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
    model.push_input_layer(dshape=[None, IMG_SIZE[0], IMG_SIZE[1], CHANNELS])
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
    model.push_input_layer(dshape=[None, IMG_SIZE[0], IMG_SIZE[1], CHANNELS])
    model.push_augment_layer(4, 4, True, True)
    # model.push_conv_layer(filter_size=[7, 7], out_channels=64, strides=[1, 1], activation='linear', has_bias=False)
    model.push_conv_layer(filter_size=[3, 3], out_channels=16, strides=[1, 1], activation='linear', has_bias=False)
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
    for i in range(7):
        model.push_res_layer([3, 3], 128, strides=[1, 1], activation='relu')
    model.push_batch_norm_layer(activation='relu')
    model.push_res_layer([3, 3], 256, strides=[2, 2], activation='relu', activate_before_residual=False)
    for i in range(2):
        model.push_res_layer([3, 3], 256, strides=[1, 1], activation='relu')
    model.push_batch_norm_layer(activation='relu')
    model.push_pool_layer('avg', kernel_size=[int(IMG_SIZE[0] / 8), int(IMG_SIZE[1] / 8)],
                          strides=[int(IMG_SIZE[0] / 8), int(IMG_SIZE[1] / 8)])
    model.push_flatten_layer()
    model.push_fully_connected_layer(NUM_LABELS, activation='linear', has_bias=True)

    return model


def model4(name=''):
    # test resnet
    model = ConvNet(name or 'ResBN')
    model.push_input_layer(dshape=[None, IMG_SIZE[0], IMG_SIZE[1], CHANNELS])
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


def eval(model, data_generator):
    loss, acc, acc3 = model.eval(model.sess, data_generator, BATCH_SIZE)
    print('[Test Set] Loss: {:.3f}, Acc: {:.2%}, Acc3: {:.2%}, eval num: {:d}'.format(
        loss, acc, acc3, data_generator.n // data_generator.batch_size * data_generator.batch_size))


def train(model, train_data_generator, valid_data_generator, batch_size, epoch):
    print("prepare training....")
    trainer = Trainer(model)
    trainer.add_regularizer('l2', 1e-3)
    trainer.set_learning_rate(update_func=
                              lambda step:
                              0.1 if step < 15 * N else
                              0.01 if step < 25 * N else
                              0.001 if step < 30 * N else 0.0001)
    trainer.set_optimizer('Momentum', 0.9)
    print("start training....")
    losses, valid_losses = trainer.train(train_data_generator, valid_data_generator, epoch*N, 500)
    model.save()
    log_step = N // batch_size // 10
    total_steps = len(losses) * log_step
    train_steps = range(0, total_steps, log_step)
    valid_steps = range(0, total_steps, log_step * 10)
    train_log_file = get_path('log', model.name_or_scope + '_train.csv')
    valid_log_file = get_path('log', model.name_or_scope + '_valid.csv')
    before_save(train_log_file)
    lists2csv(list(zip(train_steps, losses)), train_log_file, header=['step', 'loss'])
    lists2csv(list(zip(valid_steps, valid_losses)), valid_log_file, header=['step', 'loss'])


def main():
    tasks = FLAGS.task.split(',')
    if len(tasks) == 0:
        return
    init_tf_environ(gpu_num=1)
    all_data = prepare_data_fer2013(test='test' in tasks, batch_size=BATCH_SIZE)
    model = build_model(FLAGS.model, FLAGS.name)
    if 'train' in tasks:
        train(model, all_data['train'], all_data['valid'], BATCH_SIZE, FLAGS.epoch)
    else:
        model.restore()
    if 'test' in tasks:
        eval(model, all_data['test'])


if __name__ == '__main__':
    main()
