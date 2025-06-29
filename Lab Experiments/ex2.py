import cv2

# Step 1: Read the image
image_path = r"C:\Users\Heisenberg Raja\OneDrive\Documents\Padmanabhan\Slot C\Picture2.jpg"
image = cv2.imread(image_path)

# Check if image is loaded
if image is None:
    print("Error: Image not found or unable to load.")
else:
    # Step 2: Display the original image
    cv2.imshow("Original Image", image)

    # Step 3: Apply Gaussian Blur
    # Syntax: cv2.GaussianBlur(src, ksize, sigmaX)
    blurred_image = cv2.GaussianBlur(image, (15, 15), 0)

    # Step 4: Display the blurred image
    cv2.imshow("Blurred Image", blurred_image)

    # Step 5: Wait and close
    cv2.waitKey(0)
    cv2.destroyAllWindows()
