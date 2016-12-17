import tensorflow as tf
import scipy.misc
import math
import h5py
import os
import sys
import argparse
import numpy as np
import subprocess
from PIL import Image

import itertools
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [Default: 0]')
parser.add_argument('--batch', type=int, default=32, help='Batch size [Default: 32]')
parser.add_argument('model', type=str, help='Model to Load')
parser.add_argument('pretrained', type=str, help='Pretrained Model to Load')
parser.add_argument('out_dir', type=str, help='Output File')
FLAGS = parser.parse_args()

batch_size = FLAGS.batch
gpu_id = FLAGS.gpu
model_to_use = FLAGS.model
pretrained_model = FLAGS.pretrained
out_dir = FLAGS.out_dir

if not os.path.exists(out_dir):
    os.mkdir(out_dir)

model_type = __import__(model_to_use)

image_size = 48
total_cat_num = 7

print '### Using Model: ', model_to_use
print '### Batch Size: ', batch_size
print '### GPU: ', gpu_id
print '### Number of Categories: ', total_cat_num

base_dir = '/home/user/kaichun/cs221/'
dataset_dir = os.path.join(base_dir, 'data_hdf5')

all_labels_file = os.path.join(base_dir, 'all_labels.txt')
flabel = open(all_labels_file, 'r')
catid2catname = [line.rstrip() for line in flabel.readlines()]
flabel.close()

TESTING_FILE_LIST = os.path.join(dataset_dir, 'testing_file_list.txt')
TRAINING_IMAGES_MEAN_SCALE_HDF5_FILE = os.path.join(dataset_dir, 'training_images_mean_scale.h5')

def getDataFiles(list_filename):
    return [line.rstrip() for line in open(list_filename)]

def load_h5(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    return (data, label)

def load_h5_mean_scale(h5_filename):
    f = h5py.File(h5_filename)
    image_scale = f['scale'][()]
    image_mean = f['mean'][:]
    return (image_mean, image_scale)

def normalize(batch_data, mean, scale):
    return (batch_data - mean) * scale

def placeholder_inputs():
    input_ph = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size))
    label_ph = tf.placeholder(tf.int32, shape=(batch_size))
    return input_ph, label_ph

def eval():
    is_training = False

    with tf.device('/gpu:'+str(gpu_id)):
        input_ph, label_ph = placeholder_inputs()
        is_training_ph = tf.placeholder(tf.bool, shape=())
        
        pred, activation = model_type.get_model(input_ph, is_training=is_training_ph, \
                cat_num=total_cat_num, batch_size=batch_size, 
                weight_decay=0.0, bn_decay=0.0)

    saver = tf.train.Saver()
    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    
    sess = tf.Session(config=config)

    saver.restore(sess, pretrained_model)

    test_file_list = getDataFiles(TESTING_FILE_LIST)
    num_test_file = len(test_file_list)

    image_mean, image_scale = load_h5_mean_scale(TRAINING_IMAGES_MEAN_SCALE_HDF5_FILE)
    print image_mean.shape
    print image_scale

    total_seen = 0
    total_correct = 0

    acts = np.zeros((0, 256), dtype=np.float32)
    imgs = np.zeros((0, 48, 48), dtype=np.float32)
    for i in range(num_test_file):
        cur_test_filename = test_file_list[i]
        print 'Loading ', cur_test_filename

        cur_data, cur_labels = load_h5(cur_test_filename)
        cur_data = np.array(cur_data, dtype=np.float32)

        imgs = np.concatenate([imgs, cur_data], axis=0)

        cur_data = normalize(cur_data, image_mean, image_scale)

        num_data = len(cur_labels)
        num_batch = num_data / batch_size

        for j in range(num_batch):
            begidx = j * batch_size
            endidx = (j + 1) * batch_size
            feed_dict = {
                    input_ph: cur_data[begidx: endidx, ...],
                    label_ph: cur_labels[begidx: endidx],
                    is_training_ph: is_training,
                    }
            pred_val, act_val = sess.run([pred, activation], feed_dict=feed_dict)

            label_pred = np.argmax(pred_val, 1)

            acts = np.concatenate([acts, act_val], axis=0)

            total_correct += np.sum(label_pred == cur_labels[begidx: endidx])
            total_seen += batch_size

    total_acc = total_correct * 1.0 / total_seen
    print 'Accuracy: ', total_acc

    top_k = 20
    for i in range(256):
        cur_act = acts[:, i]
        idx = np.argsort(cur_act)[::-1][:top_k]
        subdir = os.path.join(out_dir, str(i))
        if not os.path.exists(subdir):
            os.mkdir(subdir)
        for j in range(top_k):
            cur_img = imgs[idx[j]]
            Image.fromarray(np.uint8(cur_img)).save(os.path.join(subdir, str(j)+'_'+str(idx[j])+'.png'))

eval()

