import cv2
import numpy as np
import matplotlib.pyplot as plt
#loading the image
image = cv2.imread('Q_4.jpg', cv2.IMREAD_GRAYSCALE)
#computing gradient magnitude using Sobel gradients
gradient_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
gradient_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
#computing gradient magnitude
gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
gradient_magnitude_stretched = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
#computing histogram of gradient magnitude
hist_gradient_magnitude = cv2.calcHist([gradient_magnitude.astype(np.uint8)],[0],None,[256],[0,256])
#computing gradient orientation (angle of gradient vector)
gradient_orientation = np.arctan2(gradient_y, gradient_x)
#converting angles to degrees
gradient_orientation_degrees = (np.degrees(gradient_orientation) + 360) % 360
#computing histogram of gradient orientation
hist_gradient_orientation = cv2.calcHist([gradient_orientation_degrees.astype(np.uint8)],[0],None,[256],[0,256])
plt.plot(hist_gradient_magnitude)
plt.title('Histogram of Gradient Magnitude')
plt.show()

plt.plot(hist_gradient_orientation)
plt.title('Histogram of Gradient Orientation')
plt.show()

plt.imshow(gradient_magnitude_stretched, cmap='gray')
plt.title('Stretched Gradient Magnitude')
plt.savefig('gradient_magnitude_stretched.jpg')
plt.imshow(gradient_orientation_degrees, cmap='hsv')
plt.title('Gradient Orientation (Degrees)')
plt.savefig('gradient_orientation_degrees.jpg')