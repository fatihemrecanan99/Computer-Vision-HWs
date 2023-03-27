import cv2
import numpy as np
import numpy as np

def dilation(img, struct_element):
    # get image dimensions
    img_h, img_w = img.shape[:2]
    # get structuring element dimensions
    str_h, str_w = struct_element.shape[:2]
    # get structuring element center coordinates
    center_h = str_h // 2
    center_w = str_w // 2
    # create empty output image
    img_dilated = np.zeros((img_h, img_w), dtype=np.uint8)
    # apply dilation operation
    for i in range(center_h, img_h - center_h):
        for j in range(center_w, img_w - center_w):
            if img[i, j] == 255:
                roi = img[i - center_h: i + center_h + 1, j - center_w: j + center_w + 1]
                if np.any(np.logical_and(roi, struct_element)):
                    img_dilated[i, j] = 255
    return img_dilated


def erosion(img, struct_element):
    # get image dimensions
    img_h, img_w = img.shape[:2]
    # get structuring element dimensions
    str_h, str_w = struct_element.shape[:2]
    # get structuring element center coordinates
    center_h = str_h // 2
    center_w = str_w // 2
    # create empty output image
    img_eroded = np.zeros((img_h, img_w), dtype=np.uint8)
    # apply erosion operation
    for i in range(center_h, img_h - center_h):
        for j in range(center_w, img_w - center_w):
            roi = img[i - center_h: i + center_h + 1, j - center_w: j + center_w + 1]
            if np.all(np.logical_and(roi, struct_element)):
                img_eroded[i, j] = 255
    return img_eroded



# Step 1: Convert image to binary by thresholding
img = cv2.imread('Figure1.jpg', cv2.IMREAD_GRAYSCALE)
threshold_value, img_binary = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY)

# Step 2: Create structuring element
structuring_element = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))

# Step 3: Apply erosion operation
img_eroded = cv2.erode(img_binary, structuring_element)

# Step 4: Apply dilation operation
img_dilated = cv2.dilate(img_eroded, structuring_element)

# Step 5: Apply erosion operation again
img_final = cv2.erode(img_dilated, structuring_element)

cv2.imshow('Original Image', img)
cv2.imshow('Binary Image', img_binary)
cv2.imshow('Eroded Image', img_eroded)
cv2.imshow('Dilated Image', img_dilated)
cv2.imshow('Final Image', img_final)
cv2.waitKey(0)
cv2.destroyAllWindows()
