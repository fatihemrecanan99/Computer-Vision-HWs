import cv2
import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt


def histogram(img_a, img_b):
    # get the number of pixels in the image
    num_pixels_a = img_a.shape[0] * img_a.shape[1]
    num_pixels_b = img_b.shape[0] * img_b.shape[1]
    # calculate the histogram
    hist_a = np.zeros((256,))
    for i in range(img_a.shape[0]):
        for j in range(img_a.shape[1]):
            hist_a[img_a[i, j]] += 1
    hist_b = np.zeros((256,))
    for k in range(img_b.shape[0]):
        for l in range(img_b.shape[1]):
            hist_b[img_b[k, l]] += 1
    # normalize the histogram
    hist_a /= num_pixels_a


    # plot the histogram as bars for a
    
    plt.bar(range(256), hist_a)
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Normalized Frequency')
    plt.title('Histogram of Grayscale Image for a')
    plt.show()

    hist_b /= num_pixels_b

    plt.bar(range(256), hist_b)
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Normalized Frequency')
    plt.title('Histogram of Grayscale Image for b')
    plt.show()
# read in the image as a grayscale image
img_a = cv2.imread('Figure2_a.jpg', cv2.IMREAD_GRAYSCALE)
img_b = cv2.imread('Figure2_b.jpg', cv2.IMREAD_GRAYSCALE)
# generate and plot the histogram
histogram(img_a,img_b)
