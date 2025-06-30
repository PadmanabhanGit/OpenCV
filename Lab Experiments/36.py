import cv2

# Step 1: Read main image and watch template
image = cv2.imread('watch.jpg')       # Image containing the watch
template = cv2.imread('watch_template.jpg') # Cropped image of just the watch

# Step 2: Convert to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

# Step 3: Apply template matching
result = cv2.matchTemplate(gray_image, gray_template, cv2.TM_CCOEFF_NORMED)

# Step 4: Get best match location
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

# Step 5: Draw rectangle around detected watch
w, h = gray_template.shape[::-1]
cv2.rectangle(image, max_loc, (max_loc[0] + w, max_loc[1] + h), (0, 255, 0), 3)

# Step 6: Show result
cv2.imshow("Detected Watch", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Optional: Save the result
cv2.imwrite("watch_detected.jpg", image)
