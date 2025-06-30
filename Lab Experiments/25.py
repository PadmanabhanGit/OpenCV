import cv2
import numpy as np

img = cv2.imread('Exp 4.jpg', 0)
kernel = np.array([[-1, -2, -1],
                   [ 0,  0,  0],
                   [ 1,  2,  1]])

gradient_sharpen = cv2.filter2D(img, -1, kernel)

cv2.imshow('Original Image', img)
cv2.imshow('Gradient Masking', gradient_sharpen)
cv2.waitKey(0)
cv2.destroyAllWindows()
