import cv2
import numpy as np

def generate_gaussian_kernel(sigma):
    size = int(2 * sigma + 1)
    kernel = np.fromfunction(
        lambda x, y: (1 / (2 * np.pi * sigma ** 2)) *
                     np.exp(-((x - size // 2) ** 2 + (y - size // 2) ** 2) /
                            (2 * sigma ** 2)),
        (size, size)
    )
    return kernel / np.sum(kernel)

def gaussian_convolve_func(input_image, sigma):
    # Load the image
    image = cv2.imread(input_image, cv2.IMREAD_GRAYSCALE)
    # Do zero padding to the image
    image = cv2.copyMakeBorder\
        (image, sigma,sigma,sigma,sigma, cv2.BORDER_CONSTANT, value=0)
    # Generate Gaussian kernel
    gaussian_kernel = generate_gaussian_kernel(sigma)
    # Get image height and width
    height, width = image.shape
    # Initialize output array
    gaussian_output = np.zeros((height, width), dtype=np.float32)
    # Perform convolution with Gaussian kernel
    for i in range(sigma, height-sigma):
        for j in range(sigma, width-sigma):
            region = image[i - sigma: i + sigma + 1,
                         j - sigma: j + sigma + 1]
            gaussian_output[i - sigma, j - sigma] \
                = np.sum(region * gaussian_kernel)

    return gaussian_output

House1_s1 = gaussian_convolve_func("House1.jpg",1)
House1_s2 = gaussian_convolve_func("House1.jpg",2)
House1_s3 = gaussian_convolve_func("House1.jpg",3)
House2_s1 = gaussian_convolve_func("House2.jpg",1)
House2_s2 = gaussian_convolve_func("House2.jpg",2)
House2_s3 = gaussian_convolve_func("House2.jpg",3)
#save results
cv2.imwrite("House1_s1.jpg",House1_s1)
cv2.imwrite("House1_s2.jpg",House1_s2)
cv2.imwrite("House1_s3.jpg",House1_s3)
cv2.imwrite("House2_s1.jpg",House2_s1)
cv2.imwrite("House2_s2.jpg",House2_s2)
cv2.imwrite("House2_s3.jpg",House2_s3)