import tensorflow as tf
import json
import numpy as np
import os
import argparse
import sys
from PIL import Image

from embedding import *

colors = np.array([[0.0, 0.0, 1.0],     # Angry
                    [1.0, 0.0, 0.0],    # Disgust
                    [0.0, 1.0, 0.0],    # Fear
                    [0.7, 0.7, 0.0],    # Happy
                    [0.7, 0.0, 0.7],    # Sad
                    [0.0, 0.7, 0.7],    # Surprise
                    [0.5, 0.5, 0.5]])   # Neutral

name2id = {'Angry': 0, 'Disgust': 1, 'Fear': 2, 'Happy': 3, 'Sad': 4, \
        'Surprise': 5, 'Neutral': 6}

parser = argparse.ArgumentParser()
parser.add_argument('feats_npy', type=str, help='feats npy file')
parser.add_argument('feats_json', type=str, help='feats json file')
parser.add_argument('imgs_dir', type=str, help='imgs dir')
parser.add_argument('out_png', type=str, help='Output Tsne Png')
parser.add_argument('mode', type=str, help='mode [no_color/gt/pred/diff]')

tmp_dir = 'tmp'
if not os.path.exists(tmp_dir):
    os.mkdir(tmp_dir)

FLAGS = parser.parse_args()

img_dir = FLAGS.imgs_dir
out_png = FLAGS.out_png

feats = np.load(FLAGS.feats_npy)
print 'feats.shape = ', feats.shape

num_file = feats.shape[0]

filenames = json.load(open(FLAGS.feats_json, 'r'))
photo_list = []

color_mode = FLAGS.mode

for i in range(num_file):
    pparse = filenames[i].split('.')[0].split('_')
    color = None
    if color_mode == 'gt':
        color = colors[name2id[pparse[5]]]
    elif color_mode == 'pred':
        color = colors[name2id[pparse[3]]]
    elif color_mode == 'diff':
        if pparse[3] != pparse[5]:
            color = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    img = np.array(Image.open(os.path.join(img_dir, filenames[i]))).astype(np.float32)
    s = img.shape[0]

    out_img = np.zeros([s, s, 3], dtype=np.float32)
    out_img[:, :, 0] = img
    out_img[:, :, 1] = img
    out_img[:, :, 2] = img

    if color != None:
        for j in range(s):
            out_img[j, 0, 0] = color[0] * 255.0
            out_img[j, 0, 1] = color[1] * 255.0
            out_img[j, 0, 2] = color[2] * 255.0
            out_img[j, 1, 0] = color[0] * 255.0
            out_img[j, 1, 1] = color[1] * 255.0
            out_img[j, 1, 2] = color[2] * 255.0
            out_img[j, -1, 0] = color[0] * 255.0
            out_img[j, -1, 1] = color[1] * 255.0
            out_img[j, -1, 2] = color[2] * 255.0
            out_img[j, -2, 0] = color[0] * 255.0
            out_img[j, -2, 1] = color[1] * 255.0
            out_img[j, -2, 2] = color[2] * 255.0
            out_img[0, j, 0] = color[0] * 255.0
            out_img[0, j, 1] = color[1] * 255.0
            out_img[0, j, 2] = color[2] * 255.0
            out_img[1, j, 0] = color[0] * 255.0
            out_img[1, j, 1] = color[1] * 255.0
            out_img[1, j, 2] = color[2] * 255.0
            out_img[-1, j, 0] = color[0] * 255.0
            out_img[-1, j, 1] = color[1] * 255.0
            out_img[-1, j, 2] = color[2] * 255.0
            out_img[-2, j, 0] = color[0] * 255.0
            out_img[-2, j, 1] = color[1] * 255.0
            out_img[-2, j, 2] = color[2] * 255.0

    Image.fromarray(np.uint8(out_img)).save(os.path.join(tmp_dir, str(i)+'.png'))
    photo_list.append(os.path.join(tmp_dir, str(i)+'.png'))

coords = apply_tsne(feats)
img = embedding_to_image_grid(photo_list, coords)
img.save(out_png)
