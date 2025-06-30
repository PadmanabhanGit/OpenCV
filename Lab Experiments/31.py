import cv2
import numpy as np

# Step 1: Read the image
image = cv2.imread('Picture1.png')  # Ensure this file is in your project folder

# Step 2: Check if image is loaded
if image is None:
    print("❌ Image not found.")
else:
    # Step 3: Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Step 4: Apply binary threshold
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # Step 5: Define kernel
    kernel = np.ones((5, 5), np.uint8)

    # Step 6: Perform Opening (Erosion → Dilation)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    # Step 7: Display images
    cv2.imshow("Original Binary Image", binary)
    cv2.imshow("After Opening Operation", opening)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Optional: Save output
    cv2.imwrite('opening_result.png', opening)
