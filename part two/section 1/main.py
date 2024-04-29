import cv2
import numpy as np

def concolv_Func(input_image):
    #loading the image
    image = cv2.imread(input_image, cv2.IMREAD_GRAYSCALE)
    #do zero padding
    image = cv2.copyMakeBorder(image,1,1,1,1,cv2.BORDER_CONSTANT,value=0)
    # Defining Sobel kernels
    prewitt_kernel_x = np.array([[-1, 0, 1],[-1, 0, 1],[-1, 0, 1]])
    prewitt_kernel_y = np.array([[1, 1, 1],[0, 0, 0],[-1, -1, 1]])
    # Get image height and width
    height, width = image.shape
    # Initialize output arrays
    prewitt_x = np.zeros((height, width), dtype=np.float32)
    prewitt_y = np.zeros((height, width), dtype=np.float32)

    # Perform convolution with Prewitt kernels
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            prewitt_x[i-1, j-1] = np.sum(image[i - 1:i + 2, j - 1:j + 2] * prewitt_kernel_x)
            prewitt_y[i-1, j-1] = np.sum(image[i - 1:i + 2, j - 1:j + 2] * prewitt_kernel_y)


    output = np.sqrt(prewitt_x ** 2 + prewitt_y ** 2)
    return output
####################################################################

#convolve the first image
House1_prewitt = concolv_Func("House1.jpg")
#convolve the second image
House2_prewitt = concolv_Func("House2.jpg")
#saving the results
cv2.imwrite("House1_prewitt.jpg",House1_prewitt)
cv2.imwrite("House2_prewitt.jpg",House2_prewitt)
