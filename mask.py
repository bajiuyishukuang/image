import numpy as np
import matplotlib.pyplot as plt
import cv2


image = cv2.imread("E:/pywork/file/water_balloons.jpg")
print('this type is', type(image), 'with dimensions', image.shape)
image_copy = np.copy(image)
image_copy = cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)


# # RGB channels
# r = image_copy[:, :, 0]  # 在rgb空间提取红色
# g = image_copy[:, :, 1]  # 在rgb空间提取绿色
# b = image_copy[:, :, 2]  # 在rgb空间提取蓝色
#
# f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 10))
#
# ax1.set_title('red')
# ax1.imshow(r, cmap='gray')
#
# ax2.set_title('Green')
# ax2.imshow(g, cmap='gray')
#
# ax3.set_title('Blue')
# ax3.imshow(b, cmap='gray')
# convert from RGB to HSV
hsv = cv2.cvtColor(image_copy, cv2.COLOR_RGB2HSV)
# h = hsv[:, :, 0]  # 在hsv空间提取
# s = hsv[:, :, 1]  # 在hsv空间提取
# v = hsv[:, :, 2]  # 在hsv空间提取
#
# f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 10))
#
# ax1.set_title('hue')
# ax1.imshow(h, cmap='gray')
#
# ax2.set_title('Saturation')
# ax2.imshow(s, cmap='gray')
#
# ax3.set_title('Value')
# ax3.imshow(v, cmap='gray')

# Define our color selection criteria in HSV values
lower_hue = np.array([160, 0, 0])
upper_hue = np.array([180, 255, 255])

# Define our color selection criteria in RGB values
lower_pink = np.array([180, 0, 100])
upper_pink = np.array([255, 55, 230])

# Define the masked area in RGB space
mask_rgb = cv2.inRange(image, lower_pink, upper_pink)

# mask the image
masked_image = np.copy(image)
masked_image[mask_rgb == 0] = [0, 0, 0]

# Vizualize the mask
plt.imshow(masked_image)

# Now try HSV!

# Define the masked area in HSV space
mask_hsv = cv2.inRange(hsv, lower_hue, upper_hue)

# mask the image
masked_image = np.copy(image_copy)
masked_image[mask_hsv == 0] = [0, 0, 0]

# Vizualize the mask
plt.imshow(masked_image)

plt.show()