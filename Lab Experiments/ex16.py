import cv2

# Step 1: Read the image
image_path = r"C:\Users\Heisenberg Raja\OneDrive\Documents\Padmanabhan\Slot C\Picture2.jpg"
image = cv2.imread(image_path)

# Step 2: Check if image is loaded
if image is None:
    print("Error: Unable to load the image.")
else:
    # Step 3: Convert to Grayscale (Canny works on grayscale images)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Step 4: Apply Canny Edge Detection
    # Thresholds can be tuned; typical values: 50 and 150
    edges = cv2.Canny(gray, threshold1=50, threshold2=150)

    # Step 5: Display the original and edge-detected images
    cv2.imshow("Original Image", image)
    cv2.imshow("Canny Edge Detection", edges)

    # Step 6: Wait and close
    cv2.waitKey(0)
    cv2.destroyAllWindows()
