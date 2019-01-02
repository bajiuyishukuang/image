import numpy as np
import matplotlib.pyplot as plt
import cv2


# Read in the image
image = cv2.imread("E:/pywork/file/brain_MR.jpg")
plt.figure()
plt.imshow(image)
# Make a copy of the image
image_copy = np.copy(image)

# Change color to RGB (from BGR)
image_copy = cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)
# Convert to grayscale for filtering
gray = cv2.cvtColor(image_copy, cv2.COLOR_RGB2GRAY)

# Create a Gaussian blurred image
# (灰度图像，核大小（要是奇数），标准差）
gray_blur = cv2.GaussianBlur(gray, (9, 9), 0)

f1, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

ax1.set_title('original gray')
ax1.imshow(gray, cmap='gray')

ax2.set_title('blurred image')
ax2.imshow(gray_blur, cmap='gray')

# High-pass filter

# 3x3 sobel filters for edge detection
sobel_x = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]])


sobel_y = np.array([[-1, -2, -1],
                    [0, 0, 0],
                    [1, 2, 1]])

# Filter the original and blurred gray_scale images using filter2D
filtered = cv2.filter2D(gray, -1, sobel_x)

# 输入图像，-1表示和输入类型格式一致， 以及核矩阵
filtered_blurred = cv2.filter2D(gray_blur, -1, sobel_y)
f2, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

ax1.set_title('original gray')
ax1.imshow(filtered, cmap='gray')

ax2.set_title('blurred image')
ax2.imshow(filtered_blurred, cmap='gray')

# Create threshold that sets all the filtered pixels to white
# Above a certain threshold

f3, binary_image = cv2.threshold(filtered_blurred, 50, 255, cv2.THRESH_BINARY)


plt.show()
