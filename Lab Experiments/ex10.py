import cv2
import numpy as np

# Step 1: Read the image
image_path = r"C:\Users\Heisenberg Raja\OneDrive\Documents\Padmanabhan\Slot C\Picture1.png"
image = cv2.imread(image_path)

# Step 2: Check if image loaded
if image is None:
    print("Error: Image not found or unable to load.")
else:
    # Step 3: Define the translation (e.g., shift right by 100 px and down by 50 px)
    tx = 100  # Shift along X (right)
    ty = 50   # Shift along Y (down)

    # Step 4: Create the translation matrix
    translation_matrix = np.float32([[1, 0, tx],
                                     [0, 1, ty]])

    # Step 5: Apply affine transformation (translation)
    rows, cols = image.shape[:2]
    translated_image = cv2.warpAffine(image, translation_matrix, (cols + tx, rows + ty))

    # Step 6: Display images
    cv2.imshow("Original Image", image)
    cv2.imshow("Translated Image (Moved)", translated_image)

    # Step 7: Wait for key press and close
    cv2.waitKey(0)
    cv2.destroyAllWindows()
