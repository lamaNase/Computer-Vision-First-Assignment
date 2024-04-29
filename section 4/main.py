import cv2
import numpy as np

image = cv2.imread("image.jpg", cv2.IMREAD_GRAYSCALE)
density = 0.1
noisy_image = image.copy()
num_of_pixels = int(density * image.size)
# Add salt noise
salt_rows, salt_cols = np.random.randint(0, image.shape[0] - 1,
                                         num_of_pixels), np.random.randint(0, image.shape[1] - 1, num_of_pixels)
noisy_image[salt_rows, salt_cols] = 255
# Add pepper noise
pepper_rows, pepper_cols = np.random.randint(0, image.shape[0] - 1,
                                             num_of_pixels), np.random.randint(0, image.shape[1] - 1, num_of_pixels)
noisy_image[pepper_rows, pepper_cols] = 0
# Apply 7x7 Median filter to the noisy image
filtered_image = cv2.medianBlur(noisy_image, 7)
cv2.imwrite("noisy_image.jpg",noisy_image)
cv2.imwrite("filtered_image.jpg", filtered_image)