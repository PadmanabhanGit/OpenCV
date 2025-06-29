import cv2
import numpy as np

def compute_homography_dlt(src_pts, dst_pts):
    assert len(src_pts) == len(dst_pts) and len(src_pts) >= 4, "Need at least 4 point pairs"

    A = []
    for (x, y), (x_prime, y_prime) in zip(src_pts, dst_pts):
        A.append([-x, -y, -1,  0,  0,  0, x * x_prime, y * x_prime, x_prime])
        A.append([ 0,  0,  0, -x, -y, -1, x * y_prime, y * y_prime, y_prime])

    A = np.array(A)
    
    # Solve Ah = 0 using SVD
    U, S, Vt = np.linalg.svd(A)
    h = Vt[-1, :] / Vt[-1, -1]  # Normalize
    H = h.reshape(3, 3)
    return H

# Step 1: Read the image
image_path = r"C:\Users\Heisenberg Raja\OneDrive\Documents\Padmanabhan\Slot C\Picture1.png"
image = cv2.imread(image_path)

if image is None:
    print("Error: Image not found.")
else:
    height, width = image.shape[:2]

    # Step 2: Define 4 corresponding points
    src_pts = np.float32([[50, 50], [width - 50, 50], [50, height - 50], [width - 50, height - 50]])
    dst_pts = np.float32([[30, 100], [width - 100, 70], [100, height - 120], [width - 70, height - 70]])

    # Step 3: Compute Homography using DLT
    H_dlt = compute_homography_dlt(src_pts, dst_pts)

    # Step 4: Apply the transformation
    transformed_image = cv2.warpPerspective(image, H_dlt, (width, height))

    # Step 5: Display the result
    cv2.imshow("Original Image", image)
    cv2.imshow("DLT Transformed Image", transformed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
