import cv2
import numpy as np

img = cv2.imread('Exp 1.png', 0)
kernel = np.ones((5,5), np.uint8)
dilated = cv2.dilate(img, kernel, iterations=1)

cv2.imshow('Original Image', img)
cv2.imshow('Dilated Image', dilated)
cv2.waitKey(0)
cv2.destroyAllWindows()
