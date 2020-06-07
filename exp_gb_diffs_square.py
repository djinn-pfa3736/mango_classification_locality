import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd

import random
import sys

import pdb

size = int(sys.argv[1])
stride = int(sys.argv[2])
file_name1 = sys.argv[3]
file_name2 = sys.argv[4]
file_name3 = sys.argv[5]
file_names = (file_name1, file_name2, file_name3)

count = 0
diffs = []
a_values = []
a_count = 0
b_count = 0
c_count = 0

total_diffs = []
for file_name in file_names:

    image = cv2.imread(file_name)
    rows, cols, depth = image.shape

    for x in range(0, cols - size, stride):
        for y in range(0, rows - size, stride):

            mask = np.zeros((rows, cols))
            cv2.rectangle(mask, (y, x), (y+size, x+size), 1, -1)
            mask_ind = np.where(mask == 1, True, False)

            hist_red = np.histogram(image[mask_ind][:,2], bins=255, range=(0, 255))
            hist_green = np.histogram(image[mask_ind][:,1], bins=255, range=(0, 255))
            hist_blue = np.histogram(image[mask_ind][:,0], bins=255, range=(0, 255))

            mode_red_ind = np.where(hist_red[0] == max(hist_red[0]), True, False)
            tmp_ind = [False]
            tmp_ind.extend(mode_red_ind)
            mode_red = hist_red[1][tmp_ind][0]

            mode_green_ind = np.where(hist_green[0][0:150] == max(hist_green[0][0:150]), True, False)
            mode_green_ind = np.concatenate([mode_green_ind, np.array([False]*106)])
            mode_green = hist_green[1][mode_green_ind][0]

            mode_blue_ind = np.where(hist_blue[0][0:150] == max(hist_blue[0][0:150]), True, False)
            mode_blue_ind = np.concatenate([mode_blue_ind, np.array([False]*106)])
            mode_blue = hist_blue[1][mode_blue_ind][0]

            # pdb.set_trace()
            # print((mode_green, mode_blue))
            if(mode_green != 0.0 and mode_blue != 0.0):
                a_values.append(mode_red)

                diff_green_blue = np.abs(mode_green - mode_blue)
                diffs.append(diff_green_blue)

                if(diff_green_blue <= 6):
                    a_count += 1
                elif(6 < diff_green_blue and diff_green_blue <= 18):
                    b_count += 1
                else:
                    c_count += 1

            """
            else:
                tmp = image.copy()
                cv2.rectangle(tmp, (y, x), (y+size, x+size), (0, 255, 0), -1)
                plt.imshow(tmp)
                plt.show()

                mask = np.zeros((rows, cols))
                cv2.rectangle(mask, (y, x), (y+size, x+size), 1, -1)
                mask_ind = np.where(mask == 1, True, False)
                plt.hist(image[mask_ind], bins=255, range=(0, 255), histtype="stepfilled")
                plt.show()

            """

    print('A: ' + str(a_count))
    print('B: ' + str(b_count))
    print('C: ' + str(c_count))

    total_diffs.extend(diffs)

print(total_diffs)

df = pd.DataFrame({'Diffs': np.array(total_diffs)})
df.to_csv('diffs_square.csv')

"""
df = pd.DataFrame({'R': np.array(a_values),
                   'G-B': np.array(diffs)})
df.to_csv('RvsGB.csv')
"""

# pdb.set_trace()

# plt.hist(diffs, range=(0, 50), bins=51)
# plt.show()
