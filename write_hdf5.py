import csv
import os
import h5py
import numpy as np

input_csv_file = 'fer2013.csv'
out_dir = 'data_hdf5'

if not os.path.exists(out_dir):
    os.mkdir(out_dir)

fin = open(input_csv_file, 'rb')
reader = csv.reader(fin)

rows = [row for row in reader][1:]

h5_batch_size = 2048
size = 48

train_test_split = ['PublicTest', 'Training', 'PrivateTest']

print 'Loading data ...'
labels = np.array([np.uint8(row[0]) for row in rows]).astype(np.uint8)
images = np.array([np.reshape(np.array(row[1].split(), dtype=np.uint8), [size, size]) \
        for row in rows]).astype(np.uint8)
usages = [row[2] for row in rows]
print 'Data loaded.'

total_data = len(usages)

train_indices = [idx for idx in range(total_data) if usages[idx] == 'Training']
val_indices = [idx for idx in range(total_data) if usages[idx] == 'PublicTest']
test_indices = [idx for idx in range(total_data) if usages[idx] == 'PrivateTest']

def save_h5(h5_filename, data, label, data_dtype='uint8', label_dtype='uint8'):
    h5_fout = h5py.File(h5_filename)
    h5_fout.create_dataset('data', data=data, compression='gzip', compression_opts=4, dtype=data_dtype)
    h5_fout.create_dataset('label', data=label, compression='gzip', compression_opts=1, dtype=label_dtype)
    h5_fout.close()

def generate_hdf5(data, label, out_path):
    num_data = len(data)
    data_res = np.zeros((h5_batch_size, size, size)).astype(np.uint8)
    label_res = np.zeros((h5_batch_size)).astype(np.uint8)
    count = 0
    for idx in range(num_data):
        data_res[idx % h5_batch_size, ...] = data[idx, ...]
        label_res[idx % h5_batch_size] = label[idx]
        if (idx + 1) % h5_batch_size == 0 or idx + 1 == num_data:
            save_h5(out_path + '_' + str(count) + '.h5', data_res[:idx % h5_batch_size+1], label_res[:idx % h5_batch_size+1])
            print '\t Dumping data to ', out_path + '_' + str(count) + '.h5'
            count += 1

print 'Generating training hdf5 files ...'
generate_hdf5(images[train_indices, ...], labels[train_indices], os.path.join(out_dir, 'train'))
print 'Generating validation hdf5 files ...'
generate_hdf5(images[val_indices, ...], labels[val_indices], os.path.join(out_dir, 'val'))
print 'Generating testing hdf5 files ...'
generate_hdf5(images[test_indices, ...], labels[test_indices], os.path.join(out_dir, 'test'))

