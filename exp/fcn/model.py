import tensorflow as tf
import sys

sys.path.append('../')

import tf_util

def get_model(input_images, is_training, cat_num, batch_size, weight_decay, bn_decay):
    input_data = tf.reshape(input_images, [batch_size, -1])
    print input_data

    net = tf_util.fully_connected(input_data, 256, bn=True, is_training=is_training, \
            weight_decay=weight_decay, bn_decay=bn_decay, scope='fc1')
    net = tf_util.fully_connected(input_data, 256, bn=True, is_training=is_training, \
            weight_decay=weight_decay, bn_decay=bn_decay, scope='fc2')
    net = tf_util.fully_connected(input_data, 128, bn=True, is_training=is_training, \
            weight_decay=weight_decay, bn_decay=bn_decay, scope='fc3')
    net = tf_util.fully_connected(input_data, 64, bn=True, is_training=is_training, \
            weight_decay=weight_decay, bn_decay=bn_decay, scope='fc4')
    net = tf_util.fully_connected(input_data, cat_num, bn=False, is_training=is_training, \
            weight_decay=weight_decay, activation_fn=None, scope='fc5')

    return net
