import cv2
import numpy as np

# Step 1: Read the image
image = cv2.imread('Picture1.png')  # Make sure this file is in your working directory

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

    # Step 6: Perform Closing (Dilation → Erosion)
    closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # Step 7: Display images
    cv2.imshow("Original Binary Image", binary)
    cv2.imshow("After Closing Operation", closing)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Optional: Save output
    cv2.imwrite('closing_result.png', closing)
