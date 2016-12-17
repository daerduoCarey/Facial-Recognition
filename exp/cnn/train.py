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

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [Default: 0]')
parser.add_argument('--batch', type=int, default=32, help='Batch size [Default: 32]')
parser.add_argument('--epoch', type=int, default=50, help='Total Epoch to Train [Default: 50]')
parser.add_argument('--wd', type=float, default=0.0, help='Weight Decay [Default: 0.0]')
parser.add_argument('--verbose', type=bool, default=False, help='Verbose Mode [Default: False]')
parser.add_argument('--describe', type=str, default='', help='Describe the current experiment')
parser.add_argument('model', type=str, help='Model to Load')
FLAGS = parser.parse_args()

batch_size = FLAGS.batch
gpu_id = FLAGS.gpu
total_training_epoch = FLAGS.epoch
weight_decay = FLAGS.wd
out_suffix = FLAGS.model + '_' + FLAGS.describe
verbose = FLAGS.verbose
model_to_use = FLAGS.model

model_type = __import__(model_to_use)

image_size = 48
total_cat_num = 7

print '### Using Model: ', model_to_use
print '### Batch Size: ', batch_size
print '### GPU: ', gpu_id
print '### Number of Categories: ', total_cat_num
print '### Training epoches: ', total_training_epoch

base_dir = '/home/user/kaichun/cs221/'
dataset_dir = os.path.join(base_dir, 'data_hdf5')

log_dir = 'log_' + out_suffix
if not os.path.exists(log_dir):
    os.mkdir(log_dir)

if verbose:
    log_result_dir = os.path.join(log_dir, 'results')
    if not os.path.exists(log_result_dir):
        os.mkdir(log_result_dir)

all_labels_file = os.path.join(base_dir, 'all_labels.txt')
flabel = open(all_labels_file, 'r')
catid2catname = [line.rstrip() for line in flabel.readlines()]
flabel.close()

TRAINING_FILE_LIST = os.path.join(dataset_dir, 'training_file_list.txt')
TESTING_FILE_LIST = os.path.join(dataset_dir, 'validation_file_list.txt')
TRAINING_IMAGES_MEAN_SCALE_HDF5_FILE = os.path.join(dataset_dir, 'training_images_mean_scale.h5')

DECAY_STEP = 30000 * 5
DECAY_RATE = 0.8

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP * 2)
BN_DECAY_CLIP = 0.99

BASE_LEARNING_RATE = 0.001
LEARNING_RATE_CLIP = 1e-5
MOMENTUM = 0.9

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

def random_flip(batch_data):
    l = batch_data.shape[0]
    new_batch_data = np.array(batch_data)
    for i in range(l):
        if np.random.random() > 0.5:
            new_batch_data[i, :, :] = np.fliplr(new_batch_data[i, :, :])
    return new_batch_data

def random_translate(batch_data, shift_factor=0.1):
    l = batch_data.shape[0]
    sx = batch_data.shape[1]
    sy = batch_data.shape[2]
    new_batch_data = np.zeros((l, sx, sy), dtype=np.float32)
    for i in range(l):
        tmp = batch_data[i, :, :]
        dx = np.random.choice(np.int32(sx * shift_factor))
        dy = np.random.choice(np.int32(sy * shift_factor))

        r = np.random.random()
        new_img = np.zeros((sx, sy), dtype=np.float32)
        if r < 0.25:
            new_img[:sx-dx, :sy-dy] = tmp[dx:, dy:]
        elif r < 0.5:
            new_img[dx:, :sy-dy] = tmp[:sx-dx, dy:]
        elif r < 0.75:
            new_img[:sx-dx, dy:] = tmp[dx:, :sy-dy]
        else:
            new_img[dx:, dy:] = tmp[:sx-dx, :sy-dy]

        new_batch_data[i, :, :] = new_img
    return new_batch_data

def random_rotate(batch_data, img_mean, img_scale, rotate_ratio=10):
    new_batch_data = np.array(batch_data, dtype=np.float32)
    for i in range(new_batch_data.shape[0]):
        img = new_batch_data[i, :, :]
        img = img / img_scale + img_mean
        img.astype(np.uint8)
        rot_degree = (np.random.random() * 2 - 1) * rotate_ratio
        new_img = scipy.misc.imrotate(img, rot_degree)
        new_img.astype(np.float32)
        new_img = (new_img - img_mean) * img_scale
        new_batch_data[i, ...] = new_img
    return new_batch_data

def random_jitter(batch_data, sigma=0.01, clip=0.05):
    B, N, C = batch_data.shape
    assert(clip > 0)
    jittered_data = np.clip(sigma * np.random.randn(B, N, C), -1*clip, clip)
    jittered_data += batch_data
    return jittered_data

def placeholder_inputs():
    input_ph = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size))
    label_ph = tf.placeholder(tf.int32, shape=(batch_size))
    return input_ph, label_ph

def get_loss(pred, label):
    return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(pred, label))

def train():
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(gpu_id)):
            input_ph, label_ph = placeholder_inputs()
            is_training_ph = tf.placeholder(tf.bool, shape=())
            
            batch = tf.Variable(0, trainable=False)
            learning_rate = tf.train.exponential_decay(
                    BASE_LEARNING_RATE,
                    batch * batch_size,
                    DECAY_STEP,
                    DECAY_RATE,
                    staircase=True
                    )
            learning_rate = tf.maximum(learning_rate, LEARNING_RATE_CLIP)

            bn_momentum = tf.train.exponential_decay(
                    BN_INIT_DECAY,
                    batch * batch_size,
                    BN_DECAY_DECAY_STEP,
                    BN_DECAY_DECAY_RATE,
                    staircase=True)
            bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)

            pred = model_type.get_model(input_ph, is_training=is_training_ph, \
                    cat_num=total_cat_num, batch_size=batch_size, \
                    weight_decay=weight_decay, bn_decay=bn_decay)

            loss = get_loss(pred, label_ph)
            tf.add_to_collection('losses', loss)

            train_variables = tf.trainable_variables()
            trainer = tf.train.AdamOptimizer(learning_rate)

            total_loss_op = tf.add_n(tf.get_collection('losses'))

            train_op = trainer.minimize(total_loss_op, var_list=train_variables, global_step=batch)

    
        saver = tf.train.Saver()
        
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        
        sess = tf.Session(config=config)

        init = tf.initialize_all_variables()
        sess.run(init)

        train_file_list = getDataFiles(TRAINING_FILE_LIST)
        num_train_file = len(train_file_list)
        test_file_list = getDataFiles(TESTING_FILE_LIST)
        num_test_file = len(test_file_list)

        flog = open(os.path.join(log_dir, 'log.txt'), 'w')

        image_mean, image_scale = load_h5_mean_scale(TRAINING_IMAGES_MEAN_SCALE_HDF5_FILE)
        print image_mean.shape
        print image_scale

        # write logs to the disk
        subprocess.call('cat '+model_to_use+'.py > '+os.path.join(log_dir, 'model.py'), shell=True)
        subprocess.call('cat '+__file__+' > '+os.path.join(log_dir, 'train.py'), shell=True)

        fcmd = open(os.path.join(log_dir, 'cmd.txt'), 'w')
        fcmd.write(str(FLAGS))
        fcmd.close()

        def train_one_epoch(train_file_idx, epoch):
            is_training = True

            total_loss = 0.0
            total_correct = 0.0
            total_seen = 0

            for i in range(num_train_file):
                cur_train_filename = train_file_list[train_file_idx[i]]
                printout(flog, 'Loading train file ' + cur_train_filename)

                cur_data, cur_labels = load_h5(cur_train_filename)
                cur_data = np.array(cur_data, dtype=np.float32)

                cur_data = normalize(cur_data, image_mean, image_scale)
                
                # Data Augmentation
                cur_data = random_flip(cur_data)
                cur_data = random_jitter(cur_data)
                cur_data = random_translate(cur_data)

                num_data = len(cur_labels)
                num_batch = num_data / batch_size

                # shuffle the training data
                order = np.arange(num_data)
                np.random.shuffle(order)

                cur_data = cur_data[order, ...]
                cur_labels = cur_labels[order]

                for j in range(num_batch):
                    begidx = j * batch_size
                    endidx = (j + 1) * batch_size
                    feed_dict = {
                            input_ph: cur_data[begidx: endidx, ...],
                            label_ph: cur_labels[begidx: endidx],
                            is_training_ph: is_training,
                            }
                    _, loss_val, pred_val, lrv = sess.run([train_op, total_loss_op, pred, learning_rate], \
                            feed_dict=feed_dict)
                    label_pred = np.argmax(pred_val, 1)
                    
                    if i == 0 and j == 0:
                        print 'Learning Rate: ', lrv

                    total_correct += np.mean(label_pred == cur_labels[begidx: endidx])
                    total_loss += loss_val
                    total_seen += 1

                    if verbose and epoch % 10 == 0:
                        log_epoch_dir = os.path.join(log_result_dir, 'train_' + str(epoch))
                        if not os.path.exists(log_epoch_dir):
                            os.mkdir(log_epoch_dir)

                        for shape_idx in range(begidx, endidx):
                            gt_label = cur_labels[shape_idx]
                            pred_label = label_pred[shape_idx - begidx]

                            if not gt_label == pred_label:
                                #cur_image = cur_data[shape_idx, ...] / image_scale + image_mean
                                cur_image = cur_data[shape_idx, ...] * 255.0 + 128
                                info = '_gt_' + catid2catname[gt_label] + '_pred_' + catid2catname[pred_label] + '_'
                                Image.fromarray(np.uint8(cur_image)).save(os.path.join(log_epoch_dir, str(train_file_idx[i])+'_'+str(shape_idx)+info+'.jpg'))

            total_loss = total_loss * 1.0 / total_seen
            total_acc = total_correct * 1.0 / total_seen

            printout(flog, 'Training Loss: %f' % total_loss)
            printout(flog, 'Training Accuracy: %f' % total_acc)

        def eval_one_epoch(epoch):
            is_training = False

            total_loss = 0.0
            total_correct = 0.0
            total_seen = 0

            total_per_cat_acc = np.zeros((total_cat_num), dtype=np.float32)
            total_per_cat_seen = np.zeros((total_cat_num), dtype=np.int32)

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
                    loss_val, pred_val = sess.run([total_loss_op, pred], \
                            feed_dict=feed_dict)
                    label_pred = np.argmax(pred_val, 1)
                    
                    total_correct += np.mean(label_pred == cur_labels[begidx: endidx])
                    total_loss += loss_val
                    total_seen += 1

                    for shape_idx in range(begidx, endidx):
                        gt_label = cur_labels[shape_idx]
                        pred_label = label_pred[shape_idx - begidx]
                        total_per_cat_acc[gt_label] += np.int32(pred_label == gt_label)
                        total_per_cat_seen[gt_label] += 1

                        if verbose and epoch % 10 == 0:
                            log_epoch_dir = os.path.join(log_result_dir, 'test_' + str(epoch))
                            if not os.path.exists(log_epoch_dir):
                                os.mkdir(log_epoch_dir)

                            if not gt_label == pred_label:
                                cur_image = cur_data[shape_idx, ...] / image_scale + image_mean
                                info = '_gt_' + catid2catname[gt_label] + '_pred_' + catid2catname[pred_label] + '_'
                                Image.fromarray(np.uint8(cur_image)).save(os.path.join(log_epoch_dir, str(shape_idx)+info+'.jpg'))

            total_loss = total_loss * 1.0 / total_seen
            total_acc = total_correct * 1.0 / total_seen

            for cat_idx in range(total_cat_num):
                if total_per_cat_seen[cat_idx] > 0:
                    printout(flog, '\tCategory %d (%s) number: %d' % (cat_idx, \
                            catid2catname[cat_idx], total_per_cat_seen[cat_idx]))
                    printout(flog, '\tAccruracy: %f' % (total_per_cat_acc[cat_idx] \
                            * 1.0 / total_per_cat_seen[cat_idx]))

            printout(flog, 'Testing Loss: %f' % total_loss)
            printout(flog, 'Testing Accuracy: %f' % total_acc)

        for epoch in range(total_training_epoch):
            if epoch % 1 == 0:
                printout(flog, '\n<<< Testing on the test dataset ...')
                eval_one_epoch(epoch)

            printout(flog, '\n>>> Training for the epoch %d/%d ...' % (epoch, total_training_epoch))
            
            train_file_idx = np.arange(0, len(train_file_list))
            np.random.shuffle(train_file_idx)

            train_one_epoch(train_file_idx, epoch)

            if (epoch + 1) % 10 == 0:
                cp_filename = saver.save(sess, os.path.join(log_dir, 'checkpoint_'+str(epoch)+'.ckpt'))
                printout(flog, 'Successfully store the checkpoint model into' + cp_filename)

            flog.flush()

        flog.close()

train()

