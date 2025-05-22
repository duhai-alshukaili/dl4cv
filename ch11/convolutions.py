from skimage.exposure import rescale_intensity
import numpy as np 
import argparse
import cv2

def convolve(image, K):
    # grab the dimensions of the images and the kernel
    (iH, iW) = image.shape[:2]
    (kH, kW) = K.shape[:2]


    # determin the padding
    pad = (kW - 1) // 2
    
    # allocate memory for the output image
    image = cv2.copyMakeBorder(image, pad, pad, pad, pad, 
        cv2.BORDER_REPLICATE)
    
    output = np.zeros((iH, iW), dtype="float")

    # loop over the input image, "sliding" the kerneal across
    # each (x, y) coordinate from left-to-right and top-to-bottom
    
    for y in np.arange(pad, iH + pad):
        for x in np.arange(pad, iW + pad):
            roi = image[y - pad:y + pad + 1, x - pad:x + pad + 1]

            # perform the convolutions
            k = (roi * K).sum()

            output[y - pad, x - pad] = k


    output = rescale_intensity(output, in_range=(0, 255))
    output = (output * 255).astype("uint8")

    return output

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, 
    help="path to the input image")

args = vars(ap.parse_args())

# construct average blurring kernaels
smallBlur = np.ones((7, 7), dtype="float") * (1.0 / (7 * 7))
largeBlur = np.ones((21, 21), dtype="float") * (1.0 / (21 * 21))
hugeBlur  = np.ones((43, 43), dtype="float") * (1.0 / (43 * 43))

# sharpening fliter
sharpern = np.array((
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0]), dtype="int")


# Laplacian 
laplacian = np.array((
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0]), dtype="int")

# Sobel x-axis edge
sobelX = np.array((
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]), dtype="int")


# Sobel y-axis edge
sobelY = np.array((
    [-1, -2, -1],
    [0, 0, 0],
    [1, 2, 1]), dtype="int")


# emboss
emboss = np.array((
    [-2, -1, 0],
    [-1, 1, 1],
    [0, 1, 2]), dtype="int")


# list of all the kernels we will apply
kernelBank = (
    ("small_blur", smallBlur),
    ("large_blur", largeBlur),
    ("huge_blur", hugeBlur),
    ("sharpen", sharpern),
    ("laplacian", laplacian),
    ("sobel_X", sobelX),
    ("sobel_Y", sobelY),
    ("emboss", emboss)
)

# load the input image and convert it to grey scale
image = cv2.imread(args["image"])
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

(bImage, gImage, rImage) = cv2.split(image)

# loop over the kernels
for (kernelName, K) in kernelBank:

    # apply the kernel to ech of the channels
    print("[INFO] applying {} kernel".format(kernelName))
    # convolveOutput = convolve(gray, K)
    (bConvOutput, gConvOutput, rConvOutput) = (convolve(bImage, K), convolve(gImage, K), convolve(rImage, K))
    
    # show the output image
    cv2.imshow("Original", image)
    #cv2.imshow("{} - convolve".format(kernelName), convolveOutput)
    cv2.imshow("{} - convolve".format(kernelName), cv2.merge([bConvOutput, gConvOutput, rConvOutput]))
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

