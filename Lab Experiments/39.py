import cv2
import numpy as np

img = cv2.imread('Exp 3.jpg', 0)

# Horizontal kernel
kernelx = np.array([[1, 0, -1],
                    [1, 0, -1],
                    [1, 0, -1]])

# Vertical kernel
kernely = np.array([[1, 1, 1],
                    [0, 0, 0],
                    [-1, -1, -1]])

prewittx = cv2.filter2D(img, -1, kernelx)
prewitty = cv2.filter2D(img, -1, kernely)

cv2.imshow('Original Image', img)
cv2.imshow('Prewitt X', prewittx)
cv2.imshow('Prewitt Y', prewitty)
cv2.waitKey(0)
cv2.destroyAllWindows()
