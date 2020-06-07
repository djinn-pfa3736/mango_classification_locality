import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd

import random
import sys

import pdb

file_name = sys.argv[1]
# size = int(sys.argv[2])
# stride = int(sys.argv[3])

# image = cv2.imread('./dataset/class_a/IMG_0366.JPG')
image = cv2.imread(file_name)
rows, cols, depth = image.shape

mask_green = np.where(image[:,:,1] < 150, True, False)
mask_diff_rg = np.where((image[:,:,2] - image[:,:,1]) > 40, True, False)

mask = mask_green & mask_diff_rg

masked_image = np.zeros((rows, cols, 3))
# masked_image[mask_green,:] = image[mask_green,:]
# masked_image[mask_diff_rg, :] = image[mask_diff_rg, :]
masked_image[mask, :] = image[mask, :]
diff_gb_image = masked_image[:,:,1] - masked_image[:,:,0]

mask_a = np.where(diff_gb_image <= 6, True, False)
mask_r_low = np.where(masked_image[:,:,2] <= 50, True, False)
mask_r_up = np.where(50 < masked_image[:,:,2], True, False)
mask_a = mask_a & mask_r_up

mask_b_sub0 = np.where(6 <= diff_gb_image, True, False)
mask_b_sub1 = np.where(diff_gb_image < 21, True, False)
mask_b_sub2 = mask_a & mask_r_low

mask_b = mask_b_sub0 & mask_b_sub1
mask_b_total = mask_b | mask_b_sub2

# mask_c = np.where(18 <= diff_gb_image, True, False)
mask_c_sub0 = np.where(21 <= diff_gb_image, True, False)
mask_c_sub1 = mask_b & mask_r_low
mask_c = mask_c_sub0 | mask_c_sub1

# total = len(diff_gb_image[mask_a]) + len(diff_gb_image[mask_b_total]) + len(diff_gb_image[mask_c])
total = len(diff_gb_image[mask_a]) + len(diff_gb_image[mask_b]) + len(diff_gb_image[mask_c])

print('A: ' + str(len(diff_gb_image[mask_a])/total))
print('B: ' + str(len(diff_gb_image[mask_b_total])/total))
print('C: ' + str(len(diff_gb_image[mask_c])/total))

image_a = masked_image[mask_a]

pdb.set_trace()

"""
for i in range(0, len(image_a)):
    if(image_a[i, 0] != 0.0):
        print(image_a[i, :])
"""

# pdb.set_trace()

# plt.hist(diffs, range=(0, 50), bins=51)
# plt.show()
