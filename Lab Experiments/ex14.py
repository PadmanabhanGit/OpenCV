import cv2
import numpy as np

# Step 1: Read the image
image_path = r"C:\Users\Heisenberg Raja\OneDrive\Documents\Padmanabhan\Slot C\Picture1.png"
image = cv2.imread(image_path)

# Step 2: Check if image is loaded
if image is None:
    print("Error: Image not found or unable to load.")
else:
    # Step 3: Define 4 source and 4 destination points
    height, width = image.shape[:2]
    src_pts = np.float32([[50, 50], [width - 50, 50], [50, height - 50], [width - 50, height - 50]])
    dst_pts = np.float32([[10, 100], [width - 100, 50], [100, height - 100], [width - 50, height - 50]])

    # Step 4: Compute Homography matrix
    H, status = cv2.findHomography(src_pts, dst_pts)

    # Step 5: Apply the homography transformation
    transformed_image = cv2.warpPerspective(image, H, (width, height))

    # Step 6: Show the original and transformed image
    cv2.imshow("Original Image", image)
    cv2.imshow("Homography Transformed Image", transformed_image)

    # Step 7: Wait and close
    cv2.waitKey(0)
    cv2.destroyAllWindows()
