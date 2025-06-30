import cv2

img = cv2.imread('Exp 4.jpg')
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img, '© Sakthi Manoj', (10,50), font, 1, (255,255,255), 2, cv2.LINE_AA)

cv2.imshow('Watermarked Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
