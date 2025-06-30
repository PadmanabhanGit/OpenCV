import cv2
import numpy as np

img = cv2.imread('Exp 3.jpg', 0)
A = 4
kernel = np.array([[0, -1, 0],
                   [-1, A, -1],
                   [0, -1, 0]])

high_boost = cv2.filter2D(img, -1, kernel)

cv2.imshow('Original Image', img)
cv2.imshow('High-Boost Masking', high_boost)
cv2.waitKey(0)
cv2.destroyAllWindows()
