import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd

import random
import sys

import pdb

def calc_radius(total_pixel, ratio):
    pixel_num = np.floor(total_pixel*ratio)
    radius = int(np.floor(np.sqrt(pixel_num/np.pi)))

    return radius

file_name = sys.argv[1]
sample_num = int(sys.argv[2])
ratio = float(sys.argv[3])

# image = cv2.imread('./dataset/class_a/IMG_0366.JPG')
image = cv2.imread(file_name)
rows, cols, depth = image.shape

total_pixel = rows*cols
radius = calc_radius(total_pixel, ratio)

count = 0
diffs = []
a_values = []
a_count = 0
b_count = 0
c_count = 0
while(count < sample_num):
    mask = np.zeros((rows, cols))

    center_x = random.randint(0, cols)
    center_y = random.randint(0, rows)

    mask = cv2.circle(mask, (center_x, center_y), radius, 1, -1)
    mask_ind = np.where(mask == 1, True, False)

    # plt.hist(image[mask_ind], bins=255, range=(0, 255), histtype="stepfilled")
    # plt.show()

    hist_red = np.histogram(image[mask_ind][:,2], bins=255, range=(0, 255))
    hist_green = np.histogram(image[mask_ind][:,1], bins=255, range=(0, 255))
    hist_blue = np.histogram(image[mask_ind][:,0], bins=255, range=(0, 255))

    mode_red_ind = np.where(hist_red[0] == max(hist_red[0]), True, False)
    tmp_ind = [False]
    tmp_ind.extend(mode_red_ind)
    mode_red = hist_red[1][tmp_ind][0]

    mode_green_ind = np.where(hist_green[0] == max(hist_green[0]), True, False)
    tmp_ind = [False]
    tmp_ind.extend(mode_green_ind)
    mode_green = hist_green[1][tmp_ind][0]

    mode_blue_ind = np.where(hist_blue[0] == max(hist_blue[0]), True, False)
    tmp_ind = [False]
    tmp_ind.extend(mode_blue_ind)
    mode_blue = hist_blue[1][tmp_ind][0]

    if(mode_green < 100 and mode_blue < 100):
        diff_green_blue = np.abs(mode_green - mode_blue)
        diffs.append(diff_green_blue)

        if(diff_green_blue <= 6):
            a_count += 1
        elif(6 < diff_green_blue and diff_green_blue <= 18):
            b_count += 1
        else:
            c_count += 1

        a_values.append(mode_red)

    count += 1
print('A: ' + str(a_count))
print('B: ' + str(b_count))
print('C: ' + str(c_count))

df = pd.DataFrame({'R': np.array(a_values),
                   'G-B': np.array(diffs)})

df.to_csv('RvsGB.csv')

pdb.set_trace()

# plt.hist(diffs, range=(0, 50), bins=51)
# plt.show()
