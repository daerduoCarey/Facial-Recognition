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
parser.add_argument('out_file', type=str, help='Output File')
FLAGS = parser.parse_args()

batch_size = FLAGS.batch
gpu_id = FLAGS.gpu
model_to_use = FLAGS.model
pretrained_model = FLAGS.pretrained
output_file = FLAGS.out_file
output_png = FLAGS.out_file + '.png'

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

TESTING_FILE_LIST = os.path.join(dataset_dir, 'validation_file_list.txt')
TRAINING_IMAGES_MEAN_SCALE_HDF5_FILE = os.path.join(dataset_dir, 'training_images_mean_scale.h5')

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Reds):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, '{:.2f}'.format(cm[i, j]),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')



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

def printout(flog, data):
    print data
    flog.write(data + '\n')

def normalize(batch_data, mean, scale):
    return (batch_data - mean) * scale

def placeholder_inputs():
    input_ph = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size))
    label_ph = tf.placeholder(tf.int32, shape=(batch_size))
    return input_ph, label_ph

def eval():
    is_training = False

    flog = open(output_file, 'w')

    with tf.device('/gpu:'+str(gpu_id)):
        input_ph, label_ph = placeholder_inputs()
        is_training_ph = tf.placeholder(tf.bool, shape=())
        
        pred = model_type.get_model(input_ph, is_training=is_training_ph, \
                cat_num=total_cat_num, batch_size=batch_size, 
                weight_decay=0.0, bn_decay=0.0)

    saver = tf.train.Saver()
    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    
    sess = tf.Session(config=config)

    printout(flog, 'Loading pretrained model ...')
    saver.restore(sess, pretrained_model)
    printout(flog, 'Model restored.')

    test_file_list = getDataFiles(TESTING_FILE_LIST)
    num_test_file = len(test_file_list)

    image_mean, image_scale = load_h5_mean_scale(TRAINING_IMAGES_MEAN_SCALE_HDF5_FILE)
    print image_mean.shape
    print image_scale

    total_seen = 0
    total_correct = 0

    y_pred = np.zeros((0), dtype=np.int32)
    y_gt = np.zeros((0), dtype=np.int32)
    for i in range(num_test_file):
        cur_test_filename = test_file_list[i]
        printout(flog, 'Loading test file ' + cur_test_filename)

        cur_data, cur_labels = load_h5(cur_test_filename)
        cur_data = np.array(cur_data, dtype=np.float32)

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
            pred_val = sess.run(pred, feed_dict=feed_dict)

            label_pred = np.argmax(pred_val, 1)

            y_gt = np.concatenate([y_gt, cur_labels[begidx: endidx]])
            y_pred = np.concatenate([y_pred, label_pred])
            
            total_correct += np.sum(label_pred == cur_labels[begidx: endidx])
            total_seen += batch_size

    cnf_mat = confusion_matrix(y_gt, y_pred)
    np.set_printoptions(precision=2)

    fig = plt.figure()
    plot_confusion_matrix(cnf_mat, classes=catid2catname, \
            normalize=True, title='Confusion Matrix')
    fig.savefig(output_png)

    total_acc = total_correct * 1.0 / total_seen
    printout(flog, 'Testing Accuracy: %f' % total_acc)

    flog.close()

eval()

