import numpy as np
import matplotlib.pyplot as plt
import cv2

# Read in the image
image = cv2.imread('images/sunflower.jpg')

# Change color to RGB (from BGR)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

## TODO: Define lower and upper thresholds for hysteresis
# right now the threshold is so small and low that it will pick up a lot of noise
lower = 0
upper = 50

edges = cv2.Canny(gray, lower, upper)

plt.figure(figsize=(20, 10))
plt.imshow(edges, cmap='gray')