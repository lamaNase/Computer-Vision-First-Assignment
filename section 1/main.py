import cv2
import numpy as np

image = cv2.imread("image.jpg",cv2.IMREAD_GRAYSCALE)
gama = 0.4
newImage = np.power(image/255.0,gama) * 255.0
#convert pixel values to 8-bit unsigned integers
newImage = np.uint8(newImage)
cv2.imwrite("powered_Image.jpg",newImage)
