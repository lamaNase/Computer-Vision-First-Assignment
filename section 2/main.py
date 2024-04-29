import cv2
import numpy as np

image = cv2.imread("image.jpg",cv2.IMREAD_GRAYSCALE)
mean = 0
variance = 40
segma = np.sqrt(variance)
noisy_image = image + np.random.normal(mean,segma,image.shape)
cv2.imwrite("noisy_image.jpg",noisy_image)
