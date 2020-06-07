import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys

import pdb

file_name = sys.argv[1]
image = cv2.imread(file_name)
rows, cols, depth = image.shape

"""
pdb.set_trace()

plt.hist(image[:,:,0].reshape(rows*cols))
plt.show()

plt.hist(image[:,:,1].reshape(rows*cols))
plt.show()

plt.hist(image[:,:,2].reshape(rows*cols))
plt.show()
"""

hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

print(np.sqrt(np.var(hsv[:,:,0].reshape(rows*cols))))
print(np.mean(hsv[:,:,0].reshape(rows*cols)))

print(np.sqrt(np.var(hsv[:,:,1].reshape(rows*cols))))
print(np.mean(hsv[:,:,1].reshape(rows*cols)))

print(np.sqrt(np.var(hsv[:,:,2].reshape(rows*cols))))
print(np.mean(hsv[:,:,2].reshape(rows*cols)))
