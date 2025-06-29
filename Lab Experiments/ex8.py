import cv2

# Step 1: Read the image
image_path = r"C:\Users\Heisenberg Raja\OneDrive\Documents\Padmanabhan\Slot C\Picture1.png"
image = cv2.imread(image_path)

# Step 2: Check if image is loaded
if image is None:
    print("Error: Image not found or unable to load.")
else:
    # Step 3: Get original dimensions
    height, width = image.shape[:2]
    print(f"Original Size: {width}x{height}")

    # Step 4: Resize to bigger size (e.g., 2x scale)
    bigger_image = cv2.resize(image, (width * 2, height * 2), interpolation=cv2.INTER_LINEAR)

    # Step 5: Resize to smaller size (e.g., 0.5x scale)
    smaller_image = cv2.resize(image, (width // 2, height // 2), interpolation=cv2.INTER_AREA)

    # Step 6: Display all images
    cv2.imshow("Original Image", image)
    cv2.imshow("Bigger Image (2x)", bigger_image)
    cv2.imshow("Smaller Image (0.5x)", smaller_image)

    # Step 7: Wait for key press and close
    cv2.waitKey(0)
    cv2.destroyAllWindows()
