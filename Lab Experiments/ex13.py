import cv2
import numpy as np

# Step 1: Load the video
video_path = r"C:\Users\Heisenberg Raja\OneDrive\Documents\Padmanabhan\Slot C\sample_video.mp4"
cap = cv2.VideoCapture(video_path)

# Step 2: Check if video opened successfully
if not cap.isOpened():
    print("Error: Cannot open video.")
else:
    print("Performing Perspective Transformation...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Step 3: Define 4 source and destination points for perspective transform
        h, w = frame.shape[:2]
        pts1 = np.float32([[50, 50], [w - 50, 50], [50, h - 50], [w - 50, h - 50]])
        pts2 = np.float32([[30, 100], [w - 100, 50], [50, h - 100], [w - 50, h - 50]])

        # Step 4: Get perspective transform matrix and apply it
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        warped = cv2.warpPerspective(frame, matrix, (w, h))

        # Step 5: Display the original and transformed frames
        cv2.imshow('Original Video', frame)
        cv2.imshow('Perspective Transformed Video', warped)

        # Step 6: Press 'q' to quit
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    # Step 7: Release and destroy
    cap.release()
    cv2.destroyAllWindows()
