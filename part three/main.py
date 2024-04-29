import cv2
import numpy as np

#loading the noisy images
image1 = cv2.imread('Noisyimage1.jpg', cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread('Noisyimage2.jpg', cv2.IMREAD_GRAYSCALE)
#define a 5x5 kernel for averaging
kernel = np.ones((5, 5), np.float32) / 25
#apply averaging filter
averaged_image1 = cv2.filter2D(image1, -1, kernel)
averaged_image2 = cv2.filter2D(image2, -1, kernel)
#apply median filter
median_image1 = cv2.medianBlur(image1, 5)
median_image2 = cv2.medianBlur(image2, 5)
#save results
cv2.imwrite('averaged_image1.jpg',averaged_image1)
cv2.imwrite('averaged_image2.jpg',averaged_image2)
cv2.imwrite('median_image1.jpg',median_image1)
cv2.imwrite('median_image2.jpg',median_image2)
