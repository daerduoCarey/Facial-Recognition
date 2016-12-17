import sys
import matplotlib.pyplot as plt
import numpy as np

grid_mlp = [1, 3, 5]
per_mlp = [37.03, 43.53, 41.55]
grid_cnn = [2, 4, 6]
per_cnn = [66.35, 68.97, 69.25]

breadth_cnn = [63.59, 67.94, 68.97]

fig = plt.figure()
test_handle, = plt.plot(grid_mlp, per_mlp, 'b', label='MLP')
train_handle, = plt.plot(grid_cnn, per_cnn, 'r', label='CNN')
plt.legend(handles=[train_handle, test_handle], loc=4)
plt.ylabel('Testing Accuracy (%)')
plt.xlabel('Number of Layers')
fig.savefig('mlp_cnn_depth.png')
plt.close(fig)
