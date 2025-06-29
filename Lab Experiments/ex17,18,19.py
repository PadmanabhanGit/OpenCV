import cv2
import numpy as np

# Step 1: Read the image
image_path = r"C:\Users\Heisenberg Raja\OneDrive\Documents\Padmanabhan\Slot C\Picture1.png"
image = cv2.imread(image_path)

# Step 2: Check if the image is loaded
if image is None:
    print("Error: Unable to load image.")
else:
    # Step 3: Convert to Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Step 4: Sobel X (vertical edges)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_x = cv2.convertScaleAbs(sobel_x)

    # Step 5: Sobel Y (horizontal edges)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_y = cv2.convertScaleAbs(sobel_y)

    # Step 6: Combine Sobel X and Y using bitwise OR (XY edges)
    sobel_xy = cv2.bitwise_or(sobel_x, sobel_y)

    # Step 7: Show all results
    cv2.imshow("Original Image", image)
    cv2.imshow("Sobel X - Vertical Edges", sobel_x)
    cv2.imshow("Sobel Y - Horizontal Edges", sobel_y)
    cv2.imshow("Sobel XY - Combined Edges", sobel_xy)

    # Step 8: Wait and close
    cv2.waitKey(0)
    cv2.destroyAllWindows()
