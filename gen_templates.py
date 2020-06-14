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

with open('h_a.pickle', 'rb') as f:
    h_a = pickle.load(f)

with open('h_b.pickle', 'rb') as f:
    h_b = pickle.load(f)

with open('h_c.pickle', 'rb') as f:
    h_c = pickle.load(f)

h_a[126:130, 126:130] = 0
h_b[126:130, 126:130] = 0
h_c[126:130, 126:130] = 0

temp_a = np.zeros_like(h_a)
temp_a_ratio = np.zeros_like(h_a)
# abc = h_a/np.max(h_a) - h_b/np.max(h_b) - h_c/np.max(h_c)
# a_ind = np.where(abc > 0.0, True, False)
ab = h_a/np.max(h_a) - h_b/np.max(h_b)
a_ind = np.where(ab > 0.0, True, False)
temp_a[a_ind] = 1.0
# temp_a_ratio[a_ind] = abc[a_ind]/np.max(abc[a_ind])
temp_a_ratio[a_ind] = ab[a_ind]/np.max(ab[a_ind])
temp_a_org = h_a/np.max(h_a)

temp_b = np.zeros_like(h_b)
temp_b_ratio = np.zeros_like(h_b)
# bac = h_b/np.max(h_b) - h_a/np.max(h_a) - h_c/np.max(h_c)
# b_ind = np.where(bac > 0.0, True, False)
ba = h_b/np.max(h_b) - h_a/np.max(h_a)
b_ind = np.where(ba > 0.0, True, False)
temp_b[b_ind] = 1.0
# temp_b_ratio[b_ind] = bac[b_ind]/np.max(bac[b_ind])
temp_b_ratio[b_ind] = ba[b_ind]/np.max(ba[b_ind])
temp_b_org = h_b/np.max(h_b)

temp_c = np.zeros_like(h_c)
temp_c_ratio = np.zeros_like(h_c)
cba = h_c/np.max(h_c) - h_b/np.max(h_b) - h_a/np.max(h_a)
c_ind = np.where(cba > 0, True, False)
temp_c[c_ind] = 1.0
temp_c_ratio[c_ind] = cba[c_ind]/np.max(cba[c_ind])
temp_c_org = h_c/np.max(h_c)

with open('temp_a.pickle', 'wb') as f:
    pickle.dump(temp_a, f)

with open('temp_b.pickle', 'wb') as f:
    pickle.dump(temp_b, f)

with open('temp_c.pickle', 'wb') as f:
    pickle.dump(temp_c, f)

with open('temp_a_ratio.pickle', 'wb') as f:
    pickle.dump(temp_a_ratio, f)

with open('temp_b_ratio.pickle', 'wb') as f:
    pickle.dump(temp_b_ratio, f)

with open('temp_c_ratio.pickle', 'wb') as f:
    pickle.dump(temp_c_ratio, f)

with open('temp_a_org.pickle', 'wb') as f:
    pickle.dump(temp_a_org, f)

with open('temp_b_org.pickle', 'wb') as f:
    pickle.dump(temp_b_org, f)

with open('temp_c_org.pickle', 'wb') as f:
    pickle.dump(temp_c_org, f)

plt.imshow(temp_a, interpolation='nearest', cmap=cm.jet)
plt.show()

plt.imshow(temp_b, interpolation='nearest', cmap=cm.jet)
plt.show()

plt.imshow(temp_c, interpolation='nearest', cmap=cm.jet)
plt.show()

plt.imshow(temp_a_ratio, interpolation='nearest', cmap=cm.jet)
plt.show()

plt.imshow(temp_b_ratio, interpolation='nearest', cmap=cm.jet)
plt.show()

plt.imshow(temp_c_ratio, interpolation='nearest', cmap=cm.jet)
plt.show()

plt.imshow(temp_a_org, interpolation='nearest', cmap=cm.jet)
plt.show()

plt.imshow(temp_b_org, interpolation='nearest', cmap=cm.jet)
plt.show()

plt.imshow(temp_c_org, interpolation='nearest', cmap=cm.jet)
plt.show()

# pdb.set_trace()
