import cv2
import numpy as np

def myImageFilter(input_image,kernal_size):
    #loading the image
    image = cv2.imread(input_image, cv2.IMREAD_GRAYSCALE)
    #define the Averaging kernal
    kernal = np.ones((kernal_size,kernal_size), np.float32) / (kernal_size*kernal_size)
    # Do the convolution with the kernel using padding
    height, width = image.shape
    output = np.zeros((height, width), dtype=np.float32)
    padding = kernal_size // 2
    for i in range(padding, height - padding):
        for j in range(padding, width - padding):
            region = image[i - padding:i + padding + 1
            , j - padding:j + padding + 1]
            output[i - padding, j - padding] = np.sum(region * kernal)

    return output.astype(np.uint8)

#convolve the first image
House1_3x3 = myImageFilter("House1.jpg",3)
House1_5x5 = myImageFilter("House1.jpg",5)
#convolve the second image
House2_3x3 = myImageFilter("House2.jpg",3)
House2_5x5 = myImageFilter("House2.jpg",5)
#saving the results
cv2.imwrite("House1_3x3.jpg",House1_3x3)
cv2.imwrite('House1_5x5.jpg',House1_5x5)
cv2.imwrite("House2_3x3.jpg",House2_3x3)
cv2.imwrite("House2_5x5.jpg",House2_5x5)
