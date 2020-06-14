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
temp_mode = int(sys.argv[3])

files = glob.glob(dir_name + '/IMG_*.JPG')
# files = glob.glob(dir_name + '/*_shrinked.JPG')
files = np.sort(files)


if(temp_mode == 1):
    with open('temp_a.pickle', 'rb') as f:
        temp_a = pickle.load(f)

    with open('temp_b.pickle', 'rb') as f:
        temp_b = pickle.load(f)

    with open('temp_c.pickle', 'rb') as f:
        temp_c = pickle.load(f)
elif(temp_mode == 2):
    with open('temp_a_ratio.pickle', 'rb') as f:
        temp_a = pickle.load(f)

    with open('temp_b_ratio.pickle', 'rb') as f:
        temp_b = pickle.load(f)

    with open('temp_c_ratio.pickle', 'rb') as f:
        temp_c = pickle.load(f)
elif(temp_mode == 3):
    with open('temp_a_org.pickle', 'rb') as f:
        temp_a = pickle.load(f)

    with open('temp_b_org.pickle', 'rb') as f:
        temp_b = pickle.load(f)

    with open('temp_c_org.pickle', 'rb') as f:
        temp_c = pickle.load(f)

a_vec = []
b_vec = []
c_vec = []
a_vec_median = []
b_vec_median = []
c_vec_median = []
a_vec_mean = []
b_vec_mean = []
c_vec_mean = []
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

    # image_lab = cv2.cvtColor(image_sub, cv2.COLOR_BGR2Lab)
    image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    L, a, b = cv2.split(image_lab)

    a = a.reshape((rows*cols, 1))[:,0]
    b = b.reshape((rows*cols, 1))[:,0]

    fig = plt.figure()
    ax = fig.add_subplot(111)

    h, xedges, yedges, image = ax.hist2d(a, b, bins=255, range=[[0, 255], [0, 255]], cmap=cm.jet)
    plt.close(fig)

    a_map = temp_a*h
    b_map = temp_b*h
    c_map = temp_c*h

    """
    a_score = np.sum(temp_a*h)/np.sum(temp_a)
    b_score = np.sum(temp_b*h)/np.sum(temp_b)
    c_score = np.sum(temp_c*h)/np.sum(temp_c)
    """

    a_ind = np.where(a_map > 0.0, True, False)
    b_ind = np.where(b_map > 0.0, True, False)
    c_ind = np.where(c_map > 0.0, True, False)

    a_score = np.sum(a_map)
    b_score = np.sum(b_map)
    c_score = np.sum(c_map)

    a_score_median = np.median(a_map[a_ind])
    b_score_median = np.median(b_map[b_ind])
    c_score_median = np.median(c_map[c_ind])

    a_score_mean = np.mean(a_map[a_ind])
    b_score_mean = np.mean(b_map[b_ind])
    c_score_mean = np.mean(c_map[c_ind])

    total = a_score + b_score + c_score
    total_median = a_score_median + b_score_median + c_score_median
    total_mean = a_score_mean + b_score_mean + c_score_mean

    print(tmp + '[SUM]: (' + str(a_score/total) + ',' + str(b_score/total) + ',' + str(c_score/total) + ')')
    print(tmp + '[MED]: (' + str(a_score_median/total_median) + ',' + str(b_score_median/total_median) + ',' + str(c_score_median/total_median) + ')')
    print(tmp + '[MEA]: (' + str(a_score_mean/total_mean) + ',' + str(b_score_mean/total_mean) + ',' + str(c_score_mean/total_mean) + ')')

    a_vec.append(a_score/total)
    b_vec.append(b_score/total)
    c_vec.append(c_score/total)

    a_vec_median.append(a_score_median/total_median)
    b_vec_median.append(b_score_median/total_median)
    c_vec_median.append(c_score_median/total_median)

    a_vec_mean.append(a_score_mean/total_mean)
    b_vec_mean.append(b_score_mean/total_mean)
    c_vec_mean.append(c_score_mean/total_mean)

df = pd.DataFrame({ 'a': a_vec,
                    'b': b_vec,
                    'c': c_vec})

df.to_csv('lab_' + class_id + '.csv')

df_median = pd.DataFrame({ 'a': a_vec_median,
                           'b': b_vec_median,
                           'c': c_vec_median})

df.to_csv('lab_median_' + class_id + '.csv')

df_mean = pd.DataFrame({ 'a': a_vec_mean,
                         'b': b_vec_mean,
                         'c': c_vec_mean})

df.to_csv('lab_mean_' + class_id + '.csv')
