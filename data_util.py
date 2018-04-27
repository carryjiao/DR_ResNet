import tensorflow as tf
import os
import numpy as np
from PIL import Image
import csv
import time



def get_labels(labels_dir):
    labels = []
    with open(labels_dir, mode='r', encoding='utf-8') as csvfile:
        label_reader = csv.reader(csvfile)
        for row in label_reader:
            labels.append(row[0])
    return labels

def make_data_set(OP_FLAG, size):
    labels_dir = 'D:/Data/DR_data/'+ OP_FLAG +'DataLabels/'+ OP_FLAG +'Labels.csv'
    labels = get_labels(labels_dir)
    bestnum = 1000
    picnum = 0
    recodefilenum = 1
    data_path = 'D:/Data/DR_data/'+ OP_FLAG +'_crop_'+ size +'/'
    write_path = 'D:/Data/DR_data/'+ OP_FLAG +'_data_'+ size +'/'
    tfrecordfilename = OP_FLAG +'_'+ size + '.tfrecords-%.3d' % recodefilenum
    writer = tf.python_io.TFRecordWriter(write_path+tfrecordfilename)
    length = len(labels)

    for index, file in enumerate(os.listdir(data_path)):
        if file.endswith(".jpeg"):
            file_path = data_path + file
            img = Image.open(file_path)

            if img.mode == "RGB":
                picnum = picnum + 1
                if picnum > bestnum:
                    picnum = 1
                    recodefilenum = recodefilenum + 1
                    tfrecordfilename = 'train_' + size + '.tfrecords-%.3d' % recodefilenum
                    writer = tf.python_io.TFRecordWriter(write_path + tfrecordfilename)

                img_raw = img.tobytes()
                example = tf.train.Example(
                    features=tf.train.Features(
                        feature={
                            "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[int(labels[index])])),
                            "img_raw": tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
                        }
                    )
                )
                writer.write(example.SerializeToString())

        if index % 100 == 0:
            print(time.strftime('%Y-%m-%d %H:%M:%S  ', time.localtime(time.time()))
                  + '处理进度：%.2f %%' % ((index / length) * 100) + '   index = %d' % index)

    writer.close()


def read_data(OP_FLAG, size, batch_size, capacity, order):


    file_path = 'C:/Data/DR_data/'+ OP_FLAG +'_data_' + size + '/'+ OP_FLAG +'_' + size + '.tfrecords-'+order
    data_files = tf.gfile.Glob(file_path)
    filename_queue = tf.train.string_input_producer(data_files, shuffle=True)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw': tf.FixedLenFeature([], tf.string),
                                       })  # 取出包含image和label的feature对象
    # tf.decode_raw可以将字符串解析成图像对应的像素数组
    image = tf.decode_raw(features['img_raw'], tf.uint8)
    image = tf.reshape(image, [int(size), int(size), 3])
    image = tf.image.per_image_standardization(image)   #图片标准化

    label = tf.cast(features['label'], tf.int32)

    # input_queue = tf.train.slice_input_producer([image, label])

    image_batch, label_batch = tf.train.batch([image, label],
                                              batch_size=batch_size,
                                              num_threads=64,
                                              capacity=capacity)
    label_batch = tf.reshape(label_batch, [batch_size])
    image_batch = tf.cast(image_batch, tf.float32)

    return image_batch, label_batch


# make_data_set(OP_FLAG='train', size='512')