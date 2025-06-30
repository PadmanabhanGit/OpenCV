import cv2
import numpy as np

# Step 1: Read the image
image = cv2.imread('Picture1.png')  # Ensure this file is in your project directory

# Step 2: Check if image is loaded
if image is None:
    print("❌ Image not found.")
else:
    # Step 3: Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Step 4: Define kernel
    kernel = np.ones((15, 15), np.uint8)  # Larger kernel helps isolate small bright spots

    # Step 5: Apply Top Hat (original - opening)
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)

    # Step 6: Show results
    cv2.imshow("Original Image (Grayscale)", gray)
    cv2.imshow("Top Hat Result", tophat)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Optional: Save result
    cv2.imwrite('tophat_result.png', tophat)
