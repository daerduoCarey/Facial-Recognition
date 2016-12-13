import os
import sys
import numpy as np

import itertools
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

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

classes = ['A', 'B', 'C', 'D', 'E', 'F']
y_gt = np.array([1,2,3,4,1,2,4,1,2,5,5,2,2,3,1,5,2,2,3,5,3,1,2,3,0,0,1,2,3,0,0])
y_pred = np.array([2,2,4,4,1,2,4,1,2,3,3,2,2,3,1,5,2,2,3,5,2,1,2,3,1,0,1,3,4,2,0])

cnf_mat = confusion_matrix(y_gt, y_pred)
np.set_printoptions(precision=2)

fig = plt.figure()
plot_confusion_matrix(cnf_mat, classes=classes, normalize=True, title='Confusion Matrix')
fig.savefig('test.png')

