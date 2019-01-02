import numpy as np
import matplotlib.pyplot as plt
import cv2


# Read in the image
image = cv2.imread("E:/pywork/file/images/curved_lane.jpg")

image_copy = np.copy(image)
image_copy = cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(image_copy, cv2.COLOR_RGB2GRAY)

sobel_y = np.array([[-1, -2, -1],
                   [0, 0, 0],
                   [1, 2, 1]])
# Filter the image using filter2D, which has inputs: (grayscale image,
# bit-depth, kernel)
filtered_image = cv2.filter2D(gray, -1, sobel_y)
# 输入图像，-1表示和输入类型格式一致， 以及核矩阵

retval, binary_images = cv2.threshold(filtered_image, 100, 255, cv2.THRESH_BINARY)
# 为了让输出更明显
# 目标图像，像素值下限，像素值上限 ，输出类型 （二进制）





plt.imshow(image_copy)
plt.imshow(gray, cmap='gray')
plt.imshow(filtered_image, cmap='gray')
plt.imshow(binary_images, cmap='gray')
plt.show()