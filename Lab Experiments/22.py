import cv2
import numpy as np

img = cv2.imread('Exp 2.jpg', 0)
kernel = np.array([[0, -1, 0],
                   [-1, 5, -1],
                   [0, -1, 0]])

sharpened = cv2.filter2D(img, -1, kernel)

cv2.imshow('Original Image', img)
cv2.imshow('Positive Center Laplacian', sharpened)
cv2.waitKey(0)
cv2.destroyAllWindows()
