import cv2
import numpy as np

# Load the image
image_path = r"C:\Users\Heisenberg Raja\OneDrive\Documents\Padmanabhan\Slot C\Picture1.png"
image = cv2.imread(image_path)

if image is None:
    print("Error: Unable to load image.")
else:
    # Convert to grayscale for sharpening
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Define Laplacian kernel with negative center
    laplacian_kernel = np.array([[0, 1, 0],
                                 [1, -4, 1],
                                 [0, 1, 0]])

    # Apply the Laplacian kernel to the grayscale image
    laplacian = cv2.filter2D(gray, cv2.CV_64F, laplacian_kernel)
    laplacian = cv2.convertScaleAbs(laplacian)

    # Sharpened image = original - laplacian
    sharpened = cv2.subtract(gray, laplacian)

    # Show images
    cv2.imshow("Original Grayscale", gray)
    cv2.imshow("Laplacian (Edges)", laplacian)
    cv2.imshow("Sharpened Image", sharpened)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
