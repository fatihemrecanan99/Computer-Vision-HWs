import numpy as np
import cv2
import matplotlib.pyplot as plt


def conv2d(image, filter):
    # get the dimensions of the image and filter
    image_height, image_width = image.shape
    filter_height, filter_width = filter.shape

    # get the center of the filter
    filter_center_h = filter_height // 2
    filter_center_w = filter_width // 2

    # pad the image with zeros on all sides
    padded_image = np.zeros((image_height + filter_height - 1, image_width + filter_width - 1))
    padded_image[filter_center_h:-filter_center_h, filter_center_w:-filter_center_w] = image

    # create an output array to hold the result of the convolution
    output = np.zeros_like(image)

    # perform the convolution
    for i in range(filter_center_h, image_height + filter_center_h):
        for j in range(filter_center_w, image_width + filter_center_w):
            patch = padded_image[i - filter_center_h:i + filter_center_h + 1,
                    j - filter_center_w:j + filter_center_w + 1]
            output[i - filter_center_h, j - filter_center_w] = np.sum(patch * filter)

    return output


# read in the image as a grayscale image
img = cv2.imread('Figure4.jpg', cv2.IMREAD_GRAYSCALE)

# define the Sobel filters
sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

# define the Prewitt filters
prewitt_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
prewitt_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])

# perform the convolution with the Sobel filters
edges_x = conv2d(img, sobel_x)
edges_y = conv2d(img, sobel_y)

# compute the magnitude of the edges
magnitude = np.sqrt(edges_x ** 2 + edges_y ** 2)

# perform the convolution with the Prewitt filters
edges_x = conv2d(img, prewitt_x)
edges_y = conv2d(img, prewitt_y)

# compute the magnitude of the edges
magnitude = np.sqrt(edges_x ** 2 + edges_y ** 2)

# display the original image and the edge detection results
plt.subplot(2, 2, 1)
plt.imshow(img, cmap='gray')
plt.title('Original')

plt.subplot(2, 2, 2)
plt.imshow(edges_x, cmap='gray')
plt.title('Sobel X')

plt.subplot(2, 2, 3)
plt.imshow(edges_y, cmap='gray')
plt.title('Sobel Y')

plt.subplot(2, 2, 4)
plt.imshow(magnitude, cmap='gray')
plt.title('Sobel Magnitude')

plt.show()
