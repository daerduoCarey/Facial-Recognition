import sys
import matplotlib.pyplot as plt
import numpy as np

input_file = sys.argv[1]
output_file = sys.argv[2]

f = open(input_file, 'r')
lines = [line.rstrip() for line in f.readlines()]
f.close()

test_loss = []
test_acc = []
train_loss = []
train_acc = []
for line in lines:
    pparse = line.split()
    for i in range(len(pparse)-1):
        if pparse[i] == 'Testing' and pparse[i+1] == 'Loss:':
            test_loss.append(np.float32(pparse[i+2]))
        if pparse[i] == 'Testing' and pparse[i+1] == 'Accuracy:':
            test_acc.append(np.float32(pparse[i+2]))
        if pparse[i] == 'Training' and pparse[i+1] == 'Loss:':
            train_loss.append(np.float32(pparse[i+2]))
        if pparse[i] == 'Training' and pparse[i+1] == 'Accuracy:':
            train_acc.append(np.float32(pparse[i+2]))

test_loss = np.array(test_loss, dtype=np.float32)
test_acc = np.array(test_acc, dtype=np.float32)
train_loss = np.array(train_loss, dtype=np.float32)
train_acc = np.array(train_acc, dtype=np.float32)

l = len(test_loss)

fig = plt.figure()
grid = np.arange(l)
test_handle, = plt.plot(grid[1:], test_loss[1:], 'b', label='Validation')
train_handle, = plt.plot(grid[1:], train_loss[1:], 'r', label='Training')
plt.legend(handles=[train_handle, test_handle])
plt.ylabel('Mean Loss')
plt.xlabel('Training Epoch')
fig.savefig(output_file+'_loss.png')
plt.close(fig)

fig = plt.figure()
grid = np.arange(l)
test_handle, = plt.plot(grid[1:], test_acc[1:], 'b', label='Validation')
train_handle, = plt.plot(grid[1:], train_acc[1:], 'r', label='Training')
plt.legend(handles=[train_handle, test_handle], loc=4)
plt.ylabel('Accuracy')
plt.xlabel('Training Epoch')
fig.savefig(output_file+'_acc.png')
plt.close(fig)
