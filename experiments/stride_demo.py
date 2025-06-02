import matplotlib.pyplot as plt
import numpy as np
import cv2
from scipy.signal import convolve2d
import argparse

# construct the argument parser and parse the argument
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, 
    help="path to the input image")
args = vars(ap.parse_args())

# Load an image and convert to grayscale
image = cv2.imread(args["image"], cv2.IMREAD_GRAYSCALE)
if image is None:
    # Use a placeholder if the image is not available
    image = np.random.randint(0, 255, (30, 20), dtype=np.uint8)

# Define a non-square kernel (e.g., 5x3)
kernel = np.array([
    [0, 0, 0],
    [0, 0, 1],
    [0, 1, 0],
    [0, 1, 1],
    [1, 0, 0]
])

# Perform valid convolution with stride (2, 2) manually
stride_y, stride_x = 2, 2
H, W = image.shape
KH, KW = kernel.shape

# Calculate output size
out_h = (H - KH) // stride_y + 1
out_w = (W - KW) // stride_x + 1
output = np.zeros((out_h, out_w), dtype=np.float32)

# Apply convolution with strides
for i in range(out_h):
    for j in range(out_w):
        region = image[i*stride_y:i*stride_y+KH, j*stride_x:j*stride_x+KW]
        output[i, j] = np.sum(region * kernel)

# output = convolve2d(image, kernel, mode="valid", boundary="fill")

# Show the original and convolved image
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(image, cmap='gray')
axes[0].set_title('Original Image')
axes[0].axis('off')

axes[1].imshow(output, cmap='gray')
axes[1].set_title('Convolved (stride=2, kernel=5x3)')
axes[1].axis('off')

print(image.shape)
print(output.shape)

plt.tight_layout()
plt.show()
