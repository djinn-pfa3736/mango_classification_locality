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

# for id in range(0, len(files)):
for file_name in files:

    # file_name = files[id]
    image = cv2.imread(file_name)
    rows, cols, depth = image.shape

    id = file_name.split('_')[0]

    r = image[:,:,2].copy()
    g = image[:,:,1].copy()
    b = image[:,:,0].copy()

    r = r.reshape(rows*cols, 1)
    g = g.reshape(rows*cols, 1)
    b = b.reshape(rows*cols, 1)

    diff_rb = (r - b)[:,0]
    diff_gb = (g - b)[:,0]

    """
    heatmap, xedges, yedges = np.histogram2d(diff_rb, diff_gb)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    plt.imshow(heatmap, extent=extent)
    plt.show()
    """

    df = pd.DataFrame({ 'R-B': diff_rb,
                        'G-B': diff_gb})

    # df.to_csv('diff_rgb.csv')

    """
    ICA = FastICA(n_components=9, random_state=0)
    df_transformed = ICA.fit_transform(df)
    A_ = ICA.mixing_.T
    """


    fig = plt.figure()
    ax = fig.add_subplot(111)

    H = ax.hist2d(df.iloc[:, 0], df.iloc[:, 1], bins=40, range=[[10, 210], [10, 90]], cmap=cm.jet)
    ax.set_title('2D Histogram')
    ax.set_xlabel('R-B')
    ax.set_ylabel('G-B')
    fig.colorbar(H[3],ax=ax)

    output = 'hist2d_' + class_id + '_' + str(id) + '.png'
    fig.savefig(output)
    # plt.show()

# pdb.set_trace()
