import cv2
import numpy as np
import matplotlib.pyplot as plt

import glob

import pdb

files_A = glob.glob("./dataset/A/*.JPG")
files_B = glob.glob("./dataset/B/*.JPG")
files_C = glob.glob("./dataset/C/*.JPG")
files_D = glob.glob("./dataset/yani/*.JPG")
# for file in files_A:
# pdb.set_trace()

L_dict = {}
a_dict = {}
b_dict = {}

for file in files_A:

    image = cv2.imread(file)
    rows, cols, depth = image.shape

    mask_green = np.where(image[:,:,1] < 150, True, False)
    mask_diff_rg = np.where((image[:,:,2] - image[:,:,1]) > 40, True, False)
    mask = mask_green & mask_diff_rg

    masked_image_Lab = np.zeros((rows, cols, depth))

    image_Lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    masked_image_Lab[mask,:] = image_Lab[mask,:]
    image_L, image_a, image_b = cv2.split(masked_image_Lab)

    image_L_val = image_L.flatten()
    nonzero_idx = np.where(image_L_val > 0, True, False)
    image_a_val = image_a.flatten()
    image_b_val = image_b.flatten()

    if 'A' in L_dict:
        L_dict['A'] = np.concatenate([L_dict['A'], image_L_val[nonzero_idx]/255*100])
        a_dict['A'] = np.concatenate([a_dict['A'], image_a_val[nonzero_idx]-128])
        b_dict['A'] = np.concatenate([b_dict['A'], image_b_val[nonzero_idx]-128])

    else:
        L_dict['A'] = image_L_val[nonzero_idx]/255*100
        a_dict['A'] = image_a_val[nonzero_idx]-128
        b_dict['A'] = image_b_val[nonzero_idx]-128


for file in files_B:

    image = cv2.imread(file)
    rows, cols, depth = image.shape

    mask_green = np.where(image[:,:,1] < 150, True, False)
    mask_diff_rg = np.where((image[:,:,2] - image[:,:,1]) > 40, True, False)
    mask = mask_green & mask_diff_rg

    masked_image_Lab = np.zeros((rows, cols, depth))

    image_Lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    masked_image_Lab[mask,:] = image_Lab[mask,:]
    image_L, image_a, image_b = cv2.split(masked_image_Lab)

    image_L_val = image_L.flatten()
    nonzero_idx = np.where(image_L_val > 0, True, False)
    image_a_val = image_a.flatten()
    image_b_val = image_b.flatten()

    if 'B' in L_dict:
        L_dict['B'] = np.concatenate([L_dict['B'], image_L_val[nonzero_idx]/255*100])
        a_dict['B'] = np.concatenate([a_dict['B'], image_a_val[nonzero_idx]-128])
        b_dict['B'] = np.concatenate([b_dict['B'], image_b_val[nonzero_idx]-128])

    else:
        L_dict['B'] = image_L_val[nonzero_idx]/255*100
        a_dict['B'] = image_a_val[nonzero_idx]-128
        b_dict['B'] = image_b_val[nonzero_idx]-128

for file in files_C:

    image = cv2.imread(file)
    rows, cols, depth = image.shape

    mask_green = np.where(image[:,:,1] < 150, True, False)
    mask_diff_rg = np.where((image[:,:,2] - image[:,:,1]) > 40, True, False)
    mask = mask_green & mask_diff_rg

    masked_image_Lab = np.zeros((rows, cols, depth))

    image_Lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    masked_image_Lab[mask,:] = image_Lab[mask,:]
    image_L, image_a, image_b = cv2.split(masked_image_Lab)

    image_L_val = image_L.flatten()
    nonzero_idx = np.where(image_L_val > 0, True, False)
    image_a_val = image_a.flatten()
    image_b_val = image_b.flatten()

    if 'C' in L_dict:
        L_dict['C'] = np.concatenate([L_dict['C'], image_L_val[nonzero_idx]/255*100])
        a_dict['C'] = np.concatenate([a_dict['C'], image_a_val[nonzero_idx]-128])
        b_dict['C'] = np.concatenate([b_dict['C'], image_b_val[nonzero_idx]-128])

    else:
        L_dict['C'] = image_L_val[nonzero_idx]/255*100
        a_dict['C'] = image_a_val[nonzero_idx]-128
        b_dict['C'] = image_b_val[nonzero_idx]-128


for file in files_D:

    image = cv2.imread(file)
    rows, cols, depth = image.shape

    mask_green = np.where(image[:,:,1] < 150, True, False)
    mask_diff_rg = np.where((image[:,:,2] - image[:,:,1]) > 40, True, False)
    mask = mask_green & mask_diff_rg

    masked_image_Lab = np.zeros((rows, cols, depth))

    image_Lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    masked_image_Lab[mask,:] = image_Lab[mask,:]
    image_L, image_a, image_b = cv2.split(masked_image_Lab)

    image_L_val = image_L.flatten()
    nonzero_idx = np.where(image_L_val > 0, True, False)
    image_a_val = image_a.flatten()
    image_b_val = image_b.flatten()

    if 'D' in L_dict:
        L_dict['D'] = np.concatenate([L_dict['D'], image_L_val[nonzero_idx]/255*100])
        a_dict['D'] = np.concatenate([a_dict['D'], image_a_val[nonzero_idx]-128])
        b_dict['D'] = np.concatenate([b_dict['D'], image_b_val[nonzero_idx]-128])

    else:
        L_dict['D'] = image_L_val[nonzero_idx]/255*100
        a_dict['D'] = image_a_val[nonzero_idx]-128
        b_dict['D'] = image_b_val[nonzero_idx]-128




pdb.set_trace()
