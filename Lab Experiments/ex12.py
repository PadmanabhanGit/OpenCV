import cv2
import numpy as np

# Step 1: Read the image
image_path = r"C:\Users\Heisenberg Raja\OneDrive\Documents\Padmanabhan\Slot C\Picture4.jpg"
image = cv2.imread(image_path)

# Step 2: Check if image is loaded
if image is None:
    print("Error: Image not found or unable to load.")
else:
    # Step 3: Define four points in the original image (source points)
    pts1 = np.float32([[50, 50], [300, 50], [50, 300], [300, 300]])

    # Step 4: Define where those four points should map to in the output image (destination points)
    pts2 = np.float32([[10, 100], [300, 50], [100, 300], [290, 290]])

    # Step 5: Get the Perspective Transform matrix
    matrix = cv2.getPerspectiveTransform(pts1, pts2)

    # Step 6: Apply the Perspective Transformation
    result = cv2.warpPerspective(image, matrix, (image.shape[1], image.shape[0]))

    # Step 7: Display the original and transformed images
    cv2.imshow("Original Image", image)
    cv2.imshow("Perspective Transformed Image", result)

    # Step 8: Wait and close
    cv2.waitKey(0)
    cv2.destroyAllWindows()
