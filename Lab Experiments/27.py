import cv2

img = cv2.imread('Exp 1.png')

# Check image size
h, w = img.shape[:2]

# Define region of interest within available size
roi = img[50:100, 50:100]  # 50×50 cropped region

# Paste it at 120,120 if possible
if h >= 170 and w >= 170:
    img[120:170, 120:170] = roi
else:
    print("Image too small for paste operation.")

cv2.imshow('Crop & Paste', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
