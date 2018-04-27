import tensorflow as tf


def inference(images, batch_size, n_classes):
    '''

    :param images: image_batch, 4D-tensor, tf.float32, [batch_size, width, height, channels]
    :param batch_size:
    :param n_classes:
    :return: logits [batch_size, n_classes]

    imges_size: w, h
    '''

    # conv1 output_size=(w,h,16)
    with tf.variable_scope('conv1') as scope:
        #定义卷积核
        weights = tf.get_variable('weights',
                                  shape=[3,3,3,16],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1,dtype=tf.float32))
                                                # 从截断的正态分布中输出随机值。
        biases = tf.get_variable('biases',
                                 shape=[16],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))

        conv = tf.nn.conv2d(images, weights, strides=[1,1,1,1], padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(pre_activation, name=scope.name)

    # pool1 and norm1  output_size=(w,h,16)
    with tf.variable_scope('pooling1_lrn') as scope:
        pool1 = tf.nn.max_pool(conv1, ksize=[1,3,3,1], strides=[1,2,2,1],
                               padding='SAME', name='pooling1')
        # 用得不多的 lrn(): local response normalization - -局部响应标准化 类似dropout,防止过拟合
        norm1 = tf.nn.lrn(pool1, depth_radius=4, bias=1.0, alpha=0.001/9.0,
                          beta=0.75, name='norm1')

    # conv2   output_size=(w,h,16)
    with tf.variable_scope('conv2') as scope:
        weights = tf.get_variable('weights',
                                  shape=[3, 3, 16, 16],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))

        biases = tf.get_variable('biases',
                                 shape=[16],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))

        conv = tf.nn.conv2d(norm1, weights, strides=[1, 1, 1, 1], padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(pre_activation, name=scope.name)

    # pool2 and norm2  output_size=(w,h,16)
    with tf.variable_scope('pooling2_lrn') as scope:
        norm2 = tf.nn.lrn(conv2, depth_radius=4, bias=1.0, alpha=0.001 / 9.0,
                          beta=0.75, name='norm2')
        pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1],
                               padding='SAME', name='pooling2')


    # local3 output_size=(batch_size, 128)
    # (batch_size, dim)* (dim, 128)
    with tf.variable_scope('local3') as scope:

        reshape = tf.reshape(pool2, shape=[batch_size, -1])  #转换为一维tensor shape=(batch_size, *)
        dim = reshape.get_shape()[1].value  #获取二维长度
        weights = tf.get_variable('weights',
                                  shape=[dim, 128],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))

        biases = tf.get_variable('biases',
                                 shape=[128],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))

        local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)


    # local4 output_size=(batch_size, 128)
    with tf.variable_scope('local4') as scope:

        weights = tf.get_variable('weights',
                                  shape=[128, 128],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))

        biases = tf.get_variable('biases',
                                 shape=[128],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))

        local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name=scope.name)


    # softmax output_size=(batch_size, n_classes)
    with tf.variable_scope('softmax_linear') as scope:

        weights = tf.get_variable('softmax_linear',
                                  shape=[128, n_classes],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))

        biases = tf.get_variable('biases',
                                 shape=[n_classes],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))

        softmax_linear = tf.add(tf.matmul(local4, weights), biases, name=scope.name)

    return softmax_linear


def losses(logits, labels):

    with tf.variable_scope('loss') as scope:
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=labels, name='xentropy_per_example')  #求交叉熵
        loss = tf.reduce_mean(cross_entropy, name='loss')  #求平均值
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
        correct = tf.nn.in_top_k(logits, labels, 1)
        correct = tf.cast(correct, tf.float16)
        accuracy = tf.reduce_mean(correct)
        tf.summary.scalar(scope.name + '/accuracy', accuracy)

    return accuracy





















