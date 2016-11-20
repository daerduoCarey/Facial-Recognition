import tensorflow as tf
import sys

sys.path.append('../')

import tf_util

def get_model(input_images, is_training, cat_num, batch_size, weight_decay, bn_decay):
    input_data = tf.reshape(input_images, [batch_size, -1])
    print input_data

    net = tf_util.dropout(input_data, is_training=is_training, keep_prob=0.5, scope='dp0')
    net = tf_util.fully_connected(net, 256, bn=True, is_training=is_training, \
            weight_decay=weight_decay, bn_decay=bn_decay, scope='fc1')
    net = tf_util.dropout(net, is_training=is_training, keep_prob=0.5, scope='dp1')

    net = tf_util.fully_connected(net, 256, bn=True, is_training=is_training, \
            weight_decay=weight_decay, bn_decay=bn_decay, scope='fc2')
    net = tf_util.dropout(net, is_training=is_training, keep_prob=0.5, scope='dp2')

    net = tf_util.fully_connected(net, 256, bn=True, is_training=is_training, \
            weight_decay=weight_decay, bn_decay=bn_decay, scope='fc3')
    net = tf_util.dropout(net, is_training=is_training, keep_prob=0.5, scope='dp3')

    net = tf_util.fully_connected(net, 128, bn=True, is_training=is_training, \
            weight_decay=weight_decay, bn_decay=bn_decay, scope='fc4')
    net = tf_util.dropout(net, is_training=is_training, keep_prob=0.5, scope='dp4')

    net = tf_util.fully_connected(net, 64, bn=True, is_training=is_training, \
            weight_decay=weight_decay, bn_decay=bn_decay, scope='fc5')
    net = tf_util.dropout(net, is_training=is_training, keep_prob=0.8, scope='dp4')

    net = tf_util.fully_connected(net, cat_num, bn=False, is_training=is_training, \
            weight_decay=weight_decay, activation_fn=None, scope='fc6')

    return net
