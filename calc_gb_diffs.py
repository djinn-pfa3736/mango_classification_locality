import numpy as np
import cv2
import matplotlib.pyplot as plt

import random
import sys
import pickle

import pdb

dir_name = sys.argv[1]
ratio = sys.argv[2]

red_file_name = dir_name + "/obj_red_" + ratio + ".pickle"
green_file_name = dir_name + "/obj_green_" + ratio + ".pickle"
blue_file_name = dir_name + "/obj_blue_" + ratio + ".pickle"

with open(red_file_name, 'rb') as f:
    results_red = pickle.load(f)

with open(green_file_name, 'rb') as f:
    results_green = pickle.load(f)

with open(blue_file_name, 'rb') as f:
    results_blue = pickle.load(f)

diffs = []
for i in range(0, len(results_red)):
    bins = np.arange(0, 255)

    mode_green_ind = np.where(results_green[i] == max(results_green[i]), True, False)
    mode_green = bins[mode_green_ind][0]

    mode_blue_ind = np.where(results_blue[i] == max(results_blue[i]), True, False)
    mode_blue = bins[mode_blue_ind][0]

    diffs.append(abs(mode_green - mode_blue))

print(diffs)
# pdb.set_trace()
