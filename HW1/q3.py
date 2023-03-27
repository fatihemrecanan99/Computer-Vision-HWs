import numpy as np
import matplotlib.pyplot as plt
import cv2

def otsu_threshold(img):
    # get the number of pixels in the image
    num_pixels = img.shape[0] * img.shape[1]

    # calculate the histogram
    hist = np.zeros((256,))
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            hist[img[i, j]] += 1

    # normalize the histogram
    hist /= num_pixels

    # calculate the cumulative sums
    cum_sum = np.cumsum(hist)

    # calculate the cumulative means
    cum_mean = np.cumsum(np.arange(256) * hist)

    # calculate the global mean
    global_mean = cum_mean[-1]

    # calculate the between-class variance for all possible thresholds
    variances = np.zeros((256,))
    for t in range(256):
        p1 = cum_sum[t]
        p2 = 1 - p1
        if p1 == 0 or p2 == 0:
            continue
        mean1 = cum_mean[t] / p1
        mean2 = (global_mean - cum_mean[t]) / p2
        variances[t] = p1 * p2 * ((mean1 - mean2) ** 2)

    # find the threshold that maximizes the between-class variance
    threshold = np.argmax(variances)

    # threshold the image
    img_binary = np.zeros(img.shape, dtype=np.uint8)
    img_binary[img >= threshold] = 255

    return img_binary


# read in the images as text files
img1 = cv2.imread('Figure3_a.jpg', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('Figure3_b.png', cv2.IMREAD_GRAYSCALE)

# apply Otsu's thresholding to the images
img_binary1 = otsu_threshold(img1)
img_binary2 = otsu_threshold(img2)

# display the input and binary images for image 1
plt.subplot(2, 2, 1)
plt.imshow(img1, cmap='gray')
plt.title('Image 1 (Input)')

plt.subplot(2, 2, 2)
plt.imshow(img_binary1, cmap='gray')
plt.title('Image 1 (Binary)')

# display the input and binary images for image 2
plt.subplot(2, 2, 3)
plt.imshow(img2, cmap='gray')
plt.title('Image 2 (Input)')

plt.subplot(2, 2, 4)
plt.imshow(img_binary2, cmap='gray')
plt.title('Image 2 (Binary)')

plt.show()
