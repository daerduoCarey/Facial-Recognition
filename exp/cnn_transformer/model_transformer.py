import tensorflow as tf
import sys

sys.path.append('../')

import tf_util
from spatial_transformer import transformer

def get_model(input_images, is_training, cat_num, batch_size, weight_decay, bn_decay):
    input_data = tf.expand_dims(input_images, -1)
    print input_data

    image_size = 48

    # local net
    net = tf_util.conv2d(input_data, 128, [3, 3], stride=[1, 1], padding='VALID', scope='stn/conv1', \
            weight_decay=weight_decay, bn_decay=bn_decay, bn=True, is_training=is_training)

    net = tf_util.conv2d(net, 256, [3, 3], stride=[1, 1], padding='VALID', scope='stn/conv2', \
            weight_decay=weight_decay, bn_decay=bn_decay, bn=True, is_training=is_training)

    net = tf_util.conv2d(net, 512, [5, 5], stride=[1, 1], padding='VALID', scope='stn/conv3', \
            weight_decay=weight_decay, bn_decay=bn_decay, bn=True, is_training=is_training)
    net = tf_util.max_pool2d(net, [2, 2], stride=[2, 2], scope='stn/mp3', padding='VALID')

    net = tf_util.conv2d(net, 512, [5, 5], stride=[1, 1], padding='VALID', scope='stn/conv4', \
            weight_decay=weight_decay, bn_decay=bn_decay, bn=True, is_training=is_training)
    net = tf_util.max_pool2d(net, [2, 2], stride=[2, 2], scope='stn/mp4', padding='VALID')

    net = tf.reshape(net, [batch_size, -1])
    print net

    net = tf_util.fully_connected(net, 256, bn=True, is_training=is_training, \
            weight_decay=weight_decay, bn_decay=bn_decay, scope='stn/fc1')

    net = tf_util.fully_connected(net, 128, bn=True, is_training=is_training, \
            weight_decay=weight_decay, bn_decay=bn_decay, scope='stn/fc2')

    net = tf_util.fully_connected(net, 6, bn=False, is_training=is_training, \
            activation_fn=None, zero_weight=True, scope='stn/fc3')

    eye = tf.constant([[1, 0, 0], [0, 1, 0]], dtype=tf.float32)
    eyes = tf.tile(tf.expand_dims(eye, 0), [batch_size, 1, 1])

    rot_mat = tf.reshape(net, [batch_size, 2, 3])
    theta = tf.add(eyes, rot_mat)
    flattened_theta = tf.reshape(theta, [batch_size, 6])
    print flattened_theta
    print input_data

    transformed_input = transformer(input_data, flattened_theta, (image_size, image_size))
    print transformed_input
    transformed_input = tf.reshape(transformed_input, [batch_size, image_size, image_size, 1])
    print transformed_input

    # main net
    net = tf_util.conv2d(transformed_input, 128, [3, 3], stride=[1, 1], padding='VALID', scope='conv1', \
            weight_decay=weight_decay, bn_decay=bn_decay, bn=True, is_training=is_training)
    print net

    net = tf_util.conv2d(net, 256, [3, 3], stride=[1, 1], padding='VALID', scope='conv2', \
            weight_decay=weight_decay, bn_decay=bn_decay, bn=True, is_training=is_training)
    print net

    net = tf_util.conv2d(net, 512, [5, 5], stride=[1, 1], padding='VALID', scope='conv3', \
            weight_decay=weight_decay, bn_decay=bn_decay, bn=True, is_training=is_training)
    net = tf_util.max_pool2d(net, [2, 2], stride=[2, 2], scope='mp3', padding='VALID')
    print net

    net = tf_util.conv2d(net, 512, [5, 5], stride=[1, 1], padding='VALID', scope='conv4', \
            weight_decay=weight_decay, bn_decay=bn_decay, bn=True, is_training=is_training)
    net = tf_util.max_pool2d(net, [2, 2], stride=[2, 2], scope='mp4', padding='VALID')
    print net

    net = tf.reshape(net, [batch_size, -1])
    print net

    net = tf_util.fully_connected(net, 256, bn=True, is_training=is_training, \
            weight_decay=weight_decay, bn_decay=bn_decay, scope='fc1')
    net = tf_util.dropout(net, is_training=is_training, keep_prob=0.5, scope='dp1')

    net = tf_util.fully_connected(net, 128, bn=True, is_training=is_training, \
            weight_decay=weight_decay, bn_decay=bn_decay, scope='fc2')
    net = tf_util.dropout(net, is_training=is_training, keep_prob=0.5, scope='dp2')

    net = tf_util.fully_connected(net, cat_num, bn=False, is_training=is_training, \
            activation_fn=None, scope='fc3')

    return net, theta, transformed_input
