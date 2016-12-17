import sys
import matplotlib.pyplot as plt
import numpy as np

breadth_cnn = [63.59, 67.94, 68.97]
ind = np.arange(3)
width = 0.3

fig, ax = plt.subplots()
ax.bar(ind + width, breadth_cnn, width, color='r')
ax.set_ylabel('Testing Accuracy')
ax.set_xticks(ind + width)
ax.set_xticklabels(('Narrow', 'Medium', 'Wide'))
fig.savefig('cnn_breadth.png')
plt.close(fig)
