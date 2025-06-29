import cv2
import numpy as np

# Step 1: Read the image
image_path = r"C:\Users\Heisenberg Raja\OneDrive\Documents\Padmanabhan\Slot C\Picture3.jpg"
image = cv2.imread(image_path)

# Step 2: Check if image is loaded
if image is None:
    print("Error: Image not found or unable to load.")
else:
    # Step 3: Define 3 points from the original image
    rows, cols = image.shape[:2]
    pts1 = np.float32([[50, 50], [200, 50], [50, 200]])

    # Step 4: Define 3 corresponding points in the output image
    pts2 = np.float32([[10, 100], [200, 50], [100, 250]])

    # Step 5: Get the Affine Transformation matrix
    matrix = cv2.getAffineTransform(pts1, pts2)

    # Step 6: Apply the Affine Transform
    result = cv2.warpAffine(image, matrix, (cols, rows))

    # Step 7: Display the original and transformed images
    cv2.imshow("Original Image", image)
    cv2.imshow("Affine Transformed Image", result)

    # Step 8: Wait and close
    cv2.waitKey(0)
    cv2.destroyAllWindows()
