import cv2
import numpy as np

# Step 1: Read the image
image_path = r"C:\Users\Heisenberg Raja\OneDrive\Documents\Padmanabhan\Slot C\Picture5.jpg"
image = cv2.imread(image_path)

# Step 2: Check if image is loaded
if image is None:
    print("Error: Image not found or unable to load.")
else:
    # Step 3: Convert image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Step 4: Create a kernel for erosion (e.g., 5x5 matrix of ones)
    kernel = np.ones((5, 5), np.uint8)

    # Step 5: Apply erosion
    eroded_image = cv2.erode(gray_image, kernel, iterations=1)

    # Step 6: Display original and eroded images
    cv2.imshow("Original Image", image)
    cv2.imshow("Eroded Image", eroded_image)

    # Step 7: Wait and close
    cv2.waitKey(0)
    cv2.destroyAllWindows()
