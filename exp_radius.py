import numpy as np
import cv2
import matplotlib.pyplot as plt

import random
import sys
import pickle

import pdb

def calc_radius(total_pixel, ratio):
    pixel_num = np.floor(total_pixel*ratio)
    radius = int(np.floor(np.sqrt(pixel_num/np.pi)))

    return radius

file_name = sys.argv[1]
sample_size = int(sys.argv[2])
sampling_num = int(sys.argv[3])

ratio_vec = [0.03, 0.05, 0.1, 0.2]

# image = cv2.imread('./dataset/class_a/IMG_0366.JPG')
image = cv2.imread(file_name)
rows, cols, depth = image.shape
total_pixel = rows*cols

for ratio in ratio_vec:
    radius = calc_radius(total_pixel, ratio)

    sampling_count = 0
    results_red = []
    results_green = []
    results_blue = []
    while(sampling_count < sampling_num):

        total_red = np.zeros((255))
        total_green = np.zeros((255))
        total_blue = np.zeros((255))
        sample_count = 0
        while(sample_count < sample_size):
            mask = np.zeros((rows, cols))

            center_x = random.randint(0, cols)
            center_y = random.randint(0, rows)

            mask = cv2.circle(mask, (center_x, center_y), radius, 1, -1)
            mask_ind = np.where(mask == 1, True, False)

            # plt.hist(image[mask_ind], bins=256, range=(0, 255), histtype="stepfilled")
            # plt.show()

            hist_red = np.histogram(image[mask_ind][:,2], bins=256, range=(0, 255))
            hist_green = np.histogram(image[mask_ind][:,1], bins=256, range=(0, 255))
            hist_blue = np.histogram(image[mask_ind][:,0], bins=256, range=(0, 255))

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
                total_red = total_red + hist_red[0]
                total_green = total_green + hist_green[0]
                total_blue = total_blue + hist_blue[0]

            sample_count += 1

        results_red.append(total_red)
        results_green.append(total_green)
        results_blue.append(total_blue)

        sampling_count += 1

    # pdb.set_trace()

    red_obj_name = "obj_red_" + str(ratio) + ".pickle"
    green_obj_name = "obj_green_" + str(ratio) + ".pickle"
    blue_obj_name = "obj_blue_" + str(ratio) + ".pickle"

    with open(red_obj_name, 'wb') as f:
        pickle.dump(results_red, f)

    with open(green_obj_name, 'wb') as f:
        pickle.dump(results_green, f)

    with open(blue_obj_name, 'wb') as f:
        pickle.dump(results_blue, f)

    for i in range(0, 10):
        red_fig_name = "hist_red_" + str(ratio) + "_sampling" + str(i) + ".png"
        green_fig_name = "hist_green_" + str(ratio) + "_sampling" + str(i) + ".png"
        blue_fig_name = "hist_blue_" + str(ratio) + "_sampling" + str(i) + ".png"

        fig = plt.figure()
        plt.bar(range(0, 255), results_red[i], color="red", linewidth=0)
        fig.savefig(red_fig_name)
        plt.close()

        fig = plt.figure()
        plt.bar(range(0, 255), results_green[i], color="green", linewidth=0)
        fig.savefig(green_fig_name)
        plt.close()

        fig = plt.figure()
        plt.bar(range(0, 255), results_blue[i], color="blue", linewidth=0)
        fig.savefig(blue_fig_name)
        plt.close()

# plt.hist(diffs, range=(0, 50), bins=51)
# plt.show()
