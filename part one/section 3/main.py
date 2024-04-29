import cv2
import numpy as np

image = cv2.imread("noisy_image.jpg",cv2.IMREAD_GRAYSCALE)
filter = np.ones((5,5),np.float32) / 25
filtered_image = cv2.filter2D(image,-1,filter)
cv2.imwrite("filtered_image.jpg",filtered_image)
