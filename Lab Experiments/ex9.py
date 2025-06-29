import cv2

# Step 1: Read the image
image_path = r"C:\Users\Heisenberg Raja\OneDrive\Documents\Padmanabhan\Slot C\Picture2.jpg"
image = cv2.imread(image_path)

# Step 2: Check if image loaded
if image is None:
    print("Error: Image not found or unable to load.")
else:
    # Step 3: Rotate 90 degrees clockwise
    rotated_cw = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

    # Step 4: Rotate 90 degrees counter-clockwise
    rotated_ccw = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # Step 5: Display the images
    cv2.imshow("Original Image", image)
    cv2.imshow("Rotated Clockwise (90°)", rotated_cw)
    cv2.imshow("Rotated Counter-Clockwise (90°)", rotated_ccw)

    # Step 6: Wait for key press and close
    cv2.waitKey(0)
    cv2.destroyAllWindows()
