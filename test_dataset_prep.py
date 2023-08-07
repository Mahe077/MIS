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
destination = '/hdd/2019CS077/Code/v0/datasets/maps/testA/'

dataA = glob.glob(datasetA)
dataA = sorted(dataA)
dataB = glob.glob(datasetB)
dataB = sorted(dataB)

count = 0

for i, j in zip(dataA, dataB):

    count += 1
    if (count > 500):
        break

    img_A = cv2.imread(j, 0)
    img_B = cv2.imread(i, 0)
    dim = (256, 256)
    img_A = cv2.resize(img_A, dim, interpolation=cv2.INTER_AREA)
    img_B = cv2.resize(img_B, dim, interpolation=cv2.INTER_AREA)

    line = img_A

    out = np.zeros((np.shape(img_B)[0]*2, np.shape(img_B)[1], 3))
    out[0:np.shape(img_B)[1], :, 0] = img_B
    out[0:np.shape(img_B)[1], :, 1] = img_B
    out[0:np.shape(img_B)[1], :, 2] = img_B
    out[np.shape(img_B)[1]:2*np.shape(img_B)[1], :, 0] = line
    out[np.shape(img_B)[1]:2*np.shape(img_B)[1], :, 1] = line
    out[np.shape(img_B)[1]:2*np.shape(img_B)[1], :, 2] = line

    # Scale the pixel values to the range 0-255
    out = (out - np.min(out)) * (255.0 / (np.max(out) - np.min(out)))

    # Convert the NumPy array to a PIL Image object
    image = Image.fromarray(out.astype(np.uint8))

    # Saving the output to the destination
    image.save(destination + str(count) + '.jpeg')
