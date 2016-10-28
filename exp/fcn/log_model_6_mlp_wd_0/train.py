import tensorflow as tf
import h5py
import os
import sys
import argparse
import numpy as np
import subprocess
import Image

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

base_dir = '/home/kaichun/projects/cs221/fer2013/'
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

DECAY_STEP = 100000
DECAY_RATE = 0.8

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP * 2)
BN_DECAY_CLIP = 0.99

BASE_LEARNING_RATE = 0.01
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

            bn_momentum = tf.train.exponential_decay(
                    BN_INIT_DECAY,
                    batch * batch_size,
                    BN_DECAY_DECAY_STEP,
                    BN_DECAY_DECAY_RATE,
                    staircase=True)
            bn_decay = tf.minimum(BN_DECAY_DECAY_STEP, 1 - bn_momentum)

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
        subprocess.call('cat model.py > '+os.path.join(log_dir, 'model.py'), shell=True)
        subprocess.call('cat '+__file__+' > '+os.path.join(log_dir, 'train.py'), shell=True)

        fcmd = open(os.path.join(log_dir, 'cmd.txt'), 'w')
        fcmd.write(str(FLAGS))
        fcmd.close()

        def train_one_epoch(train_file_idx, epoch):
            is_training = True

            for i in range(num_train_file):
                cur_train_filename = train_file_list[train_file_idx[i]]
                printout(flog, 'Loading train file ' + cur_train_filename)

                cur_data, cur_labels = load_h5(cur_train_filename)
                cur_data = normalize(cur_data, image_mean, image_scale)

                num_data = len(cur_labels)
                num_batch = num_data / batch_size

                total_loss = 0.0
                total_correct = 0.0
                total_seen = 0

                for j in range(num_batch):
                    begidx = j * batch_size
                    endidx = (j + 1) * batch_size
                    feed_dict = {
                            input_ph: cur_data[begidx: endidx, ...],
                            label_ph: cur_labels[begidx: endidx],
                            is_training_ph: is_training,
                            }
                    _, loss_val, pred_val = sess.run([train_op, total_loss_op, pred], \
                            feed_dict=feed_dict)
                    label_pred = np.argmax(pred_val, 1)
                    
                    total_correct += np.mean(label_pred == cur_labels[begidx: endidx])
                    total_loss += loss_val
                    total_seen += 1

                    if verbose and epoch % 20 == 0:
                        log_epoch_dir = os.path.join(log_result_dir, 'train_' + str(epoch))
                        if not os.path.exists(log_epoch_dir):
                            os.mkdir(log_epoch_dir)

                        for shape_idx in range(begidx, endidx):
                            gt_label = cur_labels[shape_idx]
                            pred_label = label_pred[shape_idx - begidx]

                            if not gt_label == pred_label:
                                cur_image = cur_data[shape_idx, ...] / image_scale + image_mean
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
                    _, loss_val, pred_val = sess.run([train_op, total_loss_op, pred], \
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

                        if verbose and epoch % 20 == 0:
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
            if epoch % 10 == 0:
                printout(flog, '\n<<< Testing on the test dataset ...')
                eval_one_epoch(epoch)

            printout(flog, '\n>>> Training for the epoch %d/%d ...' % (epoch, total_training_epoch))
            
            train_file_idx = np.arange(0, len(train_file_list))
            np.random.shuffle(train_file_idx)

            train_one_epoch(train_file_idx, epoch)

            if (epoch + 1) % 10 == 0:
                cp_filename = saver.save(sess, os.path.join(log_dir, 'checkpoint_.ckpt'))
                printout(flog, 'Successfully store the checkpoint model into' + cp_filename)

            flog.flush()

        flog.close()

train()

