import tensorflow as tf
import sys

sys.path.append('../')

import tf_util

def get_model(input_images, is_training, cat_num, batch_size, weight_decay, bn_decay):
    input_data = tf.expand_dims(input_images, -1)
    print input_data

    net = tf_util.conv2d(input_data, 64, [5, 5], stride=[2, 2], padding='VALID', scope='conv1', \
            weight_decay=weight_decay, bn_decay=bn_decay, bn=True, is_training=is_training)
    net = tf_util.max_pool2d(net, [2, 2], stride=[2, 2], scope='mp1', padding='VALID')
    print net

    net = tf_util.conv2d(net, 32, [3, 3], stride=[1, 1], padding='VALID', scope='conv2', \
            weight_decay=weight_decay, bn_decay=bn_decay, bn=True, is_training=is_training)
    net = tf_util.max_pool2d(net, [2, 2], stride=[2, 2], scope='mp2', padding='VALID')
    print net

    net = tf_util.conv2d(net, 32, [3, 3], stride=[1, 1], padding='VALID', scope='conv3', \
            weight_decay=weight_decay, bn_decay=bn_decay, bn=True, is_training=is_training)
    print net

    net = tf.reshape(net, [batch_size, -1])
    print net

    net = tf_util.fully_connected(net, 64, bn=True, is_training=is_training, \
            weight_decay=weight_decay, bn_decay=bn_decay, scope='fc1')
    net = tf_util.dropout(net, is_training=is_training, keep_prob=0.5, scope='dp2')

    net = tf_util.fully_connected(net, cat_num, bn=False, is_training=is_training, \
            weight_decay=weight_decay, activation_fn=None, scope='fc2')

    return net
