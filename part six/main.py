import cv2
#loading the image
image = cv2.imread('Q_4.jpg',cv2.IMREAD_GRAYSCALE)
#apply canny edge detection for diffrent threshoul values
edge1 = cv2.Canny(image,10,50*2)
edge2 = cv2.Canny(image,100,100*2)
edge3 = cv2.Canny(image,255,255)
#save the results
cv2.imwrite('edge1.jpg',edge1)
cv2.imwrite('edge2.jpg',edge2)
cv2.imwrite('edge3.jpg',edge3)
cv2.waitKey(0)
cv2.destroyAllWindows()

