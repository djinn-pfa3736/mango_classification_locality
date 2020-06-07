import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

import random
import sys

import pdb

file_name = sys.argv[1]
class_ind = sys.argv[2]
id = sys.argv[3]

image = cv2.imread(file_name)
rows, cols, depth = image.shape

r = image[:,:,2].copy()
g = image[:,:,1].copy()
b = image[:,:,0].copy()

r = r.reshape(rows*cols, 1)
g = g.reshape(rows*cols, 1)
b = b.reshape(rows*cols, 1)

df = pd.DataFrame({'R': r[:,0],
                   'G': g[:,0],
                   'B': b[:,0]})

df_sub = df.iloc[::1000, :]

# pdb.set_trace()
clust_result = linkage(df_sub, method="ward", metric="euclidean")
clusters = fcluster(clust_result, t=7, criterion="maxclust")

cluster_dict = {1:[], 2:[], 3:[], 4:[], 5:[], 6:[], 7:[]}
for i, c in enumerate(clusters):
    cluster_dict[c].append(i)

for i in range(1, 8):
    print(len(cluster_dict[i]))

    fig = plt.figure()
    plt.hist(df_sub.iloc[cluster_dict[i],0], bins=255, range=(0, 255), histtype="stepfilled", alpha=0.5)
    plt.hist(df_sub.iloc[cluster_dict[i],1], bins=255, range=(0, 255), histtype="stepfilled", alpha=0.5)
    plt.hist(df_sub.iloc[cluster_dict[i],2], bins=255, range=(0, 255), histtype="stepfilled", alpha=0.5)
    # plt.show()

    output = class_ind + '_' + id + '_' + str(len(cluster_dict[i])) + '.png'
    fig.savefig(output)
