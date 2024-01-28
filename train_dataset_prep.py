import glob as glob
from PIL import Image
from scipy import ndimage
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import cv2
import os

datasetA = '/hdd/2019CS077/Dataset/trainA1/*.JPEG'
datasetB = '/hdd/2019CS077/Dataset/trainA2/*.JPEG'
destination = '/hdd/2019CS077/Code/v0/datasets/maps/trainA/'

dataA = sorted(glob.glob(datasetA))
dataB = sorted(glob.glob(datasetB))

count = 0

for i, j in zip(dataA, dataB):

    # Limiting the dataset to only 5000 images
    count += 1
    if (count > 5000):
        break

    img_A = cv2.imread(i)
    img_B = cv2.imread(j)

    # Resizing the image and turning to grayscale
    dim = (256, 256)
    img_A = cv2.resize(img_A, dim, interpolation=cv2.INTER_LINEAR)
    img_B = cv2.resize(img_B, dim, interpolation=cv2.INTER_LINEAR)

    # Vertically stack images
    stacked_image = cv2.vconcat([img_A, img_B])

    # Save the stacked image
    # Use input image's filename for the output
    output_filename = os.path.basename(i)
    output_path = os.path.join(destination + str(count) + '.jpeg')
    cv2.imwrite(output_path, stacked_image)
print('%d images generated !' % (count-1))
