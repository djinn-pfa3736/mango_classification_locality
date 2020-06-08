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

import pdb

dir_name = sys.argv[1]
class_id = sys.argv[2]

files = glob.glob(dir_name + '/*_shrinked.JPG')

a_vec = []
b_vec = []
c_vec = []

files = np.sort(files)
# for id in range(0, len(files)):
i = 0
for file_name in files:

    # file_name = files[id]
    image = cv2.imread(file_name)
    rows, cols, depth = image.shape

    tmp = file_name.split('/')[3]
    id = tmp.split('_')[0]

    pdb.set_trace()

    image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    L, a, b = cv2.split(image_lab)

    a = a.reshape((rows*cols, 1))[:,0]
    b = b.reshape((rows*cols, 1))[:,0]
    diff = np.abs(a - b)
    # pdb.set_trace()

    fig = plt.figure()
    ax = fig.add_subplot(111)

    h, xedges, yedges, image = ax.hist2d(a, b, bins=20, range=[[120, 140], [120, 140]], cmap=cm.jet)
    # h, xedges, yedges, image = ax.hist2d(df.iloc[:, 0], df.iloc[:, 1], bins=40, cmap=cm.jet)
    ax.set_title('2D Histogram')
    ax.set_xlabel('a')
    ax.set_ylabel('b')
    fig.colorbar(image,ax=ax)

    output = 'hist2d_' + class_id + '_' + id + '_lab.png'
    fig.savefig(output)
    # plt.show()

    # pdb.set_trace()

    a_score = h[8, 8]

    # b_sub0 = h[27, 27]
    b_sub1 = h[8, 7]
    # b_sub2 = h[29, 27]
    b_sub3 = h[7, 8]
    b_sub4 = h[9, 8]
    # b_sub5 = h[27, 29]
    b_sub6 = h[8, 9]
    # b_sub7 = h[29, 29]

    b_score_median = np.median([b_sub1, b_sub3, b_sub4, b_sub6])
    b_score_mean = np.mean([b_sub1, b_sub3, b_sub4, b_sub6])
    b_score = np.sum([b_sub1, b_sub3, b_sub4, b_sub6])

    """
    c_sub0 = h[8, 6]
    c_sub1 = h[7, 7]
    c_sub2 = h[9, 7]
    c_sub3 = h[6, 8]
    c_sub4 = h[10, 8]
    c_sub5 = h[7, 9]
    c_sub6 = h[9, 9]
    c_sub7 = h[8, 10]

    c_score_median = np.median([c_sub0, c_sub1, c_sub2, c_sub3, c_sub4, c_sub5, c_sub6, c_sub7])
    c_score_mean = np.mean([c_sub0, c_sub1, c_sub2, c_sub3, c_sub4, c_sub5, c_sub6, c_sub7])
    """

    h[8, 8] = 0.0
    h[8, 7] = 0.0
    h[7, 8] = 0.0
    h[9, 8] = 0.0
    h[8, 9] = 0.0
    c_ind = np.where(h > 0.0, True, False)

    c_score_median = np.median(h[c_ind])
    c_score_mean = np.mean(h[c_ind])
    c_score = np.sum(h[c_ind])

    total = a_score + b_score + c_score

    # pdb.set_trace()
    print(tmp + ': (' + str(a_score/total) + ',' + str(b_score/total) + ',' + str(c_score/total) + ')')

    a_vec.append(a_score/total)
    b_vec.append(b_score/total)
    c_vec.append(c_score/total)

df = pd.DataFrame({ 'file': tmp,
                    'a': a_vec,
                    'b': b_vec,
                    'c': c_vec})

df.to_csv('lab.csv')



# pdb.set_trace()
