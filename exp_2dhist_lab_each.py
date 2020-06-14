import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd

from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.decomposition import FastICA

import random
import sys
import glob
import pickle

import pdb

dir_name = sys.argv[1]
class_id = sys.argv[2]

files = glob.glob(dir_name + '/IMG_*.JPG')

a_vec = []
b_vec = []
c_vec = []

files = np.sort(files)
# for id in range(0, len(files)):

h_total = np.zeros((255, 255))
for file_name in files:

    # file_name = files[id]
    image = cv2.imread(file_name)
    rows, cols, depth = image.shape

    tmp = file_name.split('/')[3]
    tmp = tmp.split('_')[1]
    id = tmp.split('.')[0]

    """
    background_sub0 = np.where((np.abs(image[:,:,2] - image[:,:,1]) < 10), True, False)
    # background_sub1 = np.where((np.abs(image[:,:,1] - image[:,:,0]) < 10), True, False)
    background_sub2 = np.where((np.abs(image[:,:,0] - image[:,:,2]) < 10), True, False)
    background_ind = np.logical_or(background_sub0, background_sub2)

    image_ind = ~(background_ind)
    image_sub = np.zeros_like(image)
    image_sub[image_ind] = image[image_ind]
    """

    image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    L, a, b = cv2.split(image_lab)

    a = a.reshape((rows*cols, 1))[:,0]
    b = b.reshape((rows*cols, 1))[:,0]
    # pdb.set_trace()

    fig = plt.figure()
    ax = fig.add_subplot(111)

    h, xedges, yedges, image = ax.hist2d(a, b, bins=255, range=[[0, 255], [0, 255]], cmap=cm.jet)
    h_total = h_total + h

    ax.set_title('2D Histogram')
    ax.set_xlabel('a')
    ax.set_ylabel('b')
    fig.colorbar(image, ax=ax)

    output = 'hist2d_' + class_id + '_' + id + '_lab.png'
    fig.savefig(output)
    plt.close(fig)
    # plt.show()

output_pickle = 'h_' + class_id + '.pickle'
with open(output_pickle, 'wb') as f:
    pickle.dump(h_total, f)

"""
plt.imshow(h_total/np.max(h_total), interpolation='nearest', cmap=cm.jet)
plt.show()
"""

# pdb.set_trace()
