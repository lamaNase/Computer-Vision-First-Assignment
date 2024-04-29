import cv2
import numpy as np

def concolv_Func(input_image):
    #loading the image
    image = cv2.imread(input_image, cv2.IMREAD_GRAYSCALE)
    #do zero padding to the image
    image = cv2.copyMakeBorder(image,1,1,1,1,cv2.BORDER_CONSTANT,value=0)
    # Defining Sobel kernels
    sobel_kernel_x = np.array([[-1, -2, -1],[0, 0, 0],[1, 2, 1]])
    sobel_kernel_y = np.array([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]])
    # Get image height and width
    height, width = image.shape
    # Initialize output arrays
    sobel_x = np.zeros((height, width), dtype=np.float32)
    sobel_y = np.zeros((height, width), dtype=np.float32)

    # Perform convolution with Sobel kernels
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            sobel_x[i-1, j-1] = np.sum(image[i - 1:i + 2, j - 1:j + 2] * sobel_kernel_x)
            sobel_y[i-1, j-1] = np.sum(image[i - 1:i + 2, j - 1:j + 2] * sobel_kernel_y)

    output = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    return output
####################################################################

#convolve the first image
House1_sobel = concolv_Func("House1.jpg")
#convolve the second image
House2_sobel = concolv_Func("House2.jpg")
#saving the results
cv2.imwrite("House1_sobel.jpg",House1_sobel)
cv2.imwrite("House2_sobel.jpg",House2_sobel)
