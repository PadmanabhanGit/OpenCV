import cv2

# Step 1: Read the image
image_path = r"C:\Users\Heisenberg Raja\OneDrive\Documents\Padmanabhan\Slot C\Picture3.jpg"
image = cv2.imread(image_path)

# Step 2: Check if image is loaded
if image is None:
    print("Error: Image not found or unable to load.")
else:
    # Step 3: Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Step 4: Apply Canny edge detection
    # Syntax: cv2.Canny(image, threshold1, threshold2)
    edges = cv2.Canny(gray_image, threshold1=100, threshold2=200)

    # Step 5: Display original and edge-detected image
    cv2.imshow("Original Image", image)
    cv2.imshow("Edge Detected (Canny)", edges)

    # Step 6: Wait for key press and close windows
    cv2.waitKey(0)
    cv2.destroyAll
