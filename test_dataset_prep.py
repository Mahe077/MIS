import glob as glob
from PIL import Image
from scipy import ndimage
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import cv2
import os


datasetA = '/hdd/2019CS077/Dataset/testA1/*.JPEG'
datasetB = '/hdd/2019CS077/Dataset/testA2/*.JPEG'
destination = '/hdd/2019CS077/TestCode/v0/datasets/maps/testA/'

dataA = glob.glob(datasetA)
dataA = sorted(dataA)
dataB = glob.glob(datasetB)
dataB = sorted(dataB)

count = 0

for i, j in zip(dataA, dataB):

    count += 1
    if (count > 500):
        break

    img_A = cv2.imread(i)
    img_B = cv2.imread(j)

    dim = (256, 256)
    img_A = cv2.resize(img_A, dim, interpolation=cv2.INTER_AREA)
    img_B = cv2.resize(img_B, dim, interpolation=cv2.INTER_AREA)

    line = img_A

    # Vertically stack images
    stacked_image = cv2.vconcat([img_A, img_B])

    # Save the stacked image
    # Use input image's filename for the output
    output_filename = os.path.basename(i)
    output_path = os.path.join(destination + str(count) + '.jpeg')
    cv2.imwrite(output_path, stacked_image)
print('%d images generated !' % (count))
