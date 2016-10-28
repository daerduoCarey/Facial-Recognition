import numpy as np
import h5py
import os

def getDataFiles(list_filename):
    return [line.rstrip() for line in open(list_filename)]

def load_h5(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    return (data, label)

def printout(flog, data):
    print data
    flog.write(data + '\n')

# main
base_dir = '/home/kaichun/projects/cs221/fer2013/'
dataset_dir = os.path.join(base_dir, 'data_hdf5')

FILE_LIST = os.path.join(dataset_dir, 'training_file_list.txt')

image_size = 48
images = np.zeros((0, image_size, image_size), dtype=np.float32)

file_list = getDataFiles(FILE_LIST)
for fn in file_list:
    data, _ = load_h5(fn)
    images = np.concatenate((images, np.float32(data)), axis=0)

image_mean = np.mean(images, axis=0)
image_scale = 1.0 / np.max(np.abs(images - image_mean))

h5_fout = h5py.File(os.path.join(dataset_dir, 'training_images_mean_scale.h5'))
h5_fout.create_dataset('mean', data=image_mean, dtype='float32')
h5_fout.create_dataset('scale', data=image_scale, dtype='float32')
h5_fout.close()
