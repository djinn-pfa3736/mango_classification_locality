import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

import random
import sys

import pdb

file_name = sys.argv[1]

image = cv2.imread(file_name)
rows, cols, depth = image.shape

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

df.to_csv('diff_rgb.csv')
