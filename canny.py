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

# Try Canny using "wide" and "tight" thresholds
wide = cv2.Canny(gray, 30, 100)
tight = cv2.Canny(gray, 200, 240)

# Display the images
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

ax1.set_title('wide')
ax1.imshow(wide, cmap='gray')

ax2.set_title('tight')
ax2.imshow(tight, cmap='gray')
