import cv2

# Step 1: Correct the path using raw string
image_path = r"C:\Users\Heisenberg Raja\OneDrive\Documents\Padmanabhan\Slot C\Picture1.png"
image = cv2.imread(image_path)

# Check if image is loaded correctly
if image is None:
    print("Error: Image not found or unable to load.")
else:
    # Step 2: Display the original image
    cv2.imshow('Original Image', image)

    # Step 3: Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Step 4: Display the grayscale image
    cv2.imshow('Grayscale Image', gray_image)

    # Step 5: Wait for a key press and then close the windows
    cv2.waitKey(0)
    cv2.destroyAllWindows()
