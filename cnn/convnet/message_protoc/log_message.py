from . import log_message_pb2 as message


def create_training_log_message(epoch, batch, batch_num, epoch_loss, batch_acc, learning_rate, time, **kwargs):
    msg = message.TrainLog()
    msg.epoch = epoch
    msg.batch = batch
    msg.batch_num = batch_num
    msg.batch_loss = epoch_loss
    msg.batch_acc = batch_acc
    msg.learning_rate = learning_rate
    msg.time = time

    return msg


def add_evaluation_log_message(eval_msg, loss, acc, acc5, time, eval_num):
    eval_msg.loss = loss
    eval_msg.acc = acc
    eval_msg.acc5 = acc5
    eval_msg.time = time
    eval_msg.eval_num = eval_num
    return eval_msg


def log_beautiful_print(train_message):
    out = '[Epoch {:>2}]'.format(train_message.epoch)
    out += '(batch{:>3}/{:>3}) '.format(train_message.batch, train_message.batch_num)
    out += 'Time: {:.1f}s, Loss: {:.3f}, Acc: {:.2%}, lr: {:.4f}'.format(train_message.time,
                                                              train_message.batch_loss,
                                                              train_message.batch_acc,
                                                              train_message.learning_rate)
    print(out)
    if train_message.HasField('eval_message'):
        eval_msg = train_message.eval_message
        temp = 'Epoch {:>2}'.format(train_message.epoch)
        out2 = '{:-^30}\n'.format('Validation: '+temp)
        out2 += 'Time: {:.2f}s, Loss: {:.3f}, Acc: {:.2%}, Acc5: {:.2%}, eval_num: {:d}'\
            .format(eval_msg.time,
                    eval_msg.loss,
                    eval_msg.acc,
                    eval_msg.acc5,
                    eval_msg.eval_num)
        # out2 += '\n{:*^30}'.format(temp + ' Done')
        print(out2)
