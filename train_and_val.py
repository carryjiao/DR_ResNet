import os
import data_util
import numpy as np
import time
import resnet
from hyper_parameters import *


N_CLASSES = 5
IMG_W = 128
IMG_H = 128
BATCH_SIZE = 64
CAPACITY = 2000
MAX_STEP = 30000
learning_rate = 0.0001

def run_training():


    logs_train_dir = 'D:/Model/DR/logs/train/'
    logs_val_dir = 'D:/Model/DR/logs/val/'

    train_batch, train_label_batch = data_util.read_data(OP_FLAG='train', size='128',
                                                   batch_size=BATCH_SIZE, capacity=CAPACITY, order='0*')

    val_batch, val_label_batch = data_util.read_data(OP_FLAG='train', size='128',
                                               batch_size=BATCH_SIZE, capacity=CAPACITY, order='1*')

    x = tf.placeholder(tf.float32, shape=[BATCH_SIZE, IMG_W, IMG_H, 3])
    y_ = tf.placeholder(tf.int32, shape=[BATCH_SIZE])

    # logits = inference(x, BATCH_SIZE, N_CLASSES)
    logits = resnet.inference(x, 5, reuse=False)
    loss = losses(logits, y_)
    accuracy = evaluation(logits, y_)
    train_op = trainning(loss, learning_rate)

    # config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    # with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
        # tf.Session(config=tf.ConfigProto(allow_growth=True))

        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        summary_op = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(logs_train_dir, sess.graph)
        val_writer = tf.summary.FileWriter(logs_val_dir, sess.graph)

        try:
            for step in np.arange(MAX_STEP):
                if coord.should_stop():
                    break

                tra_images, tra_labels = sess.run([train_batch, train_label_batch])
                _, tra_loss, tra_acc = sess.run([train_op, loss, accuracy],
                                                feed_dict={x:tra_images, y_:tra_labels})
                if step % 5 == 0:
                    print(time.strftime('%Y-%m-%d %H:%M:%S  ',time.localtime(time.time())) +
                          'Step %d, train loss = %.2f, train accuracy = %.2f%%' % (step, tra_loss, tra_acc*100.0))
                    summary_str = sess.run(summary_op, feed_dict={x:tra_images, y_:tra_labels})
                    train_writer.add_summary(summary_str, step)

                if step % 20 == 0 or (step + 1) == MAX_STEP:
                    val_images, val_labels = sess.run([val_batch, val_label_batch])
                    val_loss, val_acc = sess.run([loss, accuracy],
                                                 feed_dict={x:val_images, y_:val_labels})

                    print(time.strftime('%Y-%m-%d %H:%M:%S  ',time.localtime(time.time())) +
                          'Step %d, val loss = %.2f, val accuracy = %.2f%%' % (step, val_loss, val_acc * 100.0))
                    summary_str = sess.run(summary_op, feed_dict={x:val_images, y_:val_labels})
                    val_writer.add_summary(summary_str, step)

                if step % 2000 == 0 or (step + 1) == MAX_STEP:
                    checkpoint_path = os.path.join(logs_train_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=step)

        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached')
        finally:
            coord.request_stop()
        coord.join(threads)


def losses(logits, labels):

    with tf.variable_scope('loss') as scope:
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=labels, name='xentropy_per_example')  #求交叉熵
        regularizer = tf.contrib.layers.l2_regularizer(scale=FLAGS.weight_decay)
        reg_term = tf.contrib.layers.apply_regularization(regularizer)
        loss = tf.reduce_mean(cross_entropy, name='loss') + reg_term #求平均值
        tf.summary.scalar(scope.name + '/loss', loss)

    return loss


def trainning(loss, learning_rate):

    with tf.name_scope('optimizer'):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_op = optimizer.minimize(loss, global_step=global_step)

    return train_op


def evaluation(logits, labels):

    with tf.variable_scope('accuracy') as scope:
        # tf.nn.in_top_k() 主要是用于计算预测的结果和实际结果的是否相等，返回一个bool类型的张量
        # tf.nn.in_top_k(prediction, target,K): prediction就是表示你预测的结果，
        # 大小就是预测样本的数量乘以输出的维度，类型是tf.float32等。
        # target就是实际样本类别的标签，大小就是样本数量的个数。
        # K表示每个样本的预测结果的前K个最大的数里面是否含有target中的值。一般都是取1。
        #correct = tf.nn.in_top_k(logits, labels, 1)
        correct = tf.equal(tf.arg_max(logits, 1), tf.arg_max(labels, 1))
        correct = tf.cast(correct, tf.float32)
        accuracy = tf.reduce_mean(correct)
        tf.summary.scalar(scope.name + '/accuracy', accuracy)

    return accuracy



run_training()

