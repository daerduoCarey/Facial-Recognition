import tensorflow as tf
import sys

sys.path.append('../')

import tf_util

def get_model(input_images, is_training, cat_num, batch_size, weight_decay, bn_decay):
    input_data = tf.expand_dims(input_images, -1)
    print input_data

    net = tf_util.conv2d(input_data, 64, [3, 3], stride=[1, 1], padding='SAME', scope='conv1', \
            weight_decay=weight_decay, bn_decay=bn_decay, bn=True, is_training=is_training)
    net = tf_util.max_pool2d(net, [2, 2], stride=[2, 2], scope='mp1', padding='SAME')
    print net

    net = tf_util.conv2d(net, 128, [5, 5], stride=[1, 1], padding='SAME', scope='conv2', \
            weight_decay=weight_decay, bn_decay=bn_decay, bn=True, is_training=is_training)
    net = tf_util.max_pool2d(net, [2, 2], stride=[2, 2], scope='mp2', padding='SAME')
    print net

    net = tf_util.conv2d(net, 512, [3, 3], stride=[1, 1], padding='SAME', scope='conv3', \
            weight_decay=weight_decay, bn_decay=bn_decay, bn=True, is_training=is_training)
    net = tf_util.max_pool2d(net, [2, 2], stride=[2, 2], scope='mp3', padding='SAME')
    print net

    net = tf_util.conv2d(net, 512, [3, 3], stride=[1, 1], padding='SAME', scope='conv4', \
            weight_decay=weight_decay, bn_decay=bn_decay, bn=True, is_training=is_training)
    net = tf_util.max_pool2d(net, [2, 2], stride=[2, 2], scope='mp4', padding='SAME')
    print net

    net = tf.reshape(net, [batch_size, -1])
    print net

    net = tf_util.fully_connected(net, 256, bn=True, is_training=is_training, \
            weight_decay=weight_decay, bn_decay=bn_decay, scope='fc1')
    net = tf_util.dropout(net, is_training=is_training, keep_prob=0.5, scope='dp1')

    net = tf_util.fully_connected(net, 512, bn=True, is_training=is_training, \
            weight_decay=weight_decay, bn_decay=bn_decay, scope='fc2')
    net = tf_util.dropout(net, is_training=is_training, keep_prob=0.5, scope='dp2')

    net = tf_util.fully_connected(net, cat_num, bn=False, is_training=is_training, \
            weight_decay=weight_decay, activation_fn=None, scope='fc3')

    return net
