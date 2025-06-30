import cv2

# Step 1: Load the video
cap = cv2.VideoCapture('stock1.webm')  # Replace with your video filename

# Step 2: Read all frames and store them in a list
frames = []
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frames.append(frame)

cap.release()

# Step 3: Play frames in reverse
for frame in reversed(frames):
    cv2.imshow("Reverse Video", frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):  # Press 'q' to exit early
        break

cv2.destroyAllWindows()
