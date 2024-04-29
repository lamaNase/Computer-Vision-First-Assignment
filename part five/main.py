import cv2
#loading the images
image1 = cv2.imread('walk_1.jpg')
image2 = cv2.imread('walk_2.jpg')
#convert them to grey scale
image1_grey = cv2.cvtColor(image1,cv2.COLOR_BGR2GRAY)
image2_grey = cv2.cvtColor(image2,cv2.COLOR_BGR2GRAY)
#subtract image 2 from image 1
image_Q4 = image1_grey - image2_grey
#show the resul
cv2.imshow('Image', image_Q4)
cv2.waitKey(0)
cv2.destroyAllWindows()
#save the result image
cv2.imwrite("image_Q4.jpg",image_Q4)
