import cv2

# Step 1: Read the video file
video_path = r"C:\Users\Heisenberg Raja\OneDrive\Documents\Padmanabhan\Slot C\sample_video.mp4"
cap = cv2.VideoCapture(video_path)

# Check if video opened successfully
if not cap.isOpened():
    print("Error: Cannot open video.")
else:
    print("Playing video...")

    # Step 2: Get the original frame rate (fps)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    print("Original FPS:", fps)

    # ----- NORMAL SPEED -----
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("Normal Speed", frame)
        if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):  # Normal speed
            break

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to start

    # ----- SLOW MOTION (delay more) -----
    print("Playing in slow motion...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("Slow Motion", frame)
        if cv2.waitKey(int(1000 / (fps / 0.5))) & 0xFF == ord('q'):  # Half speed
            break

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to start

    # ----- FAST MOTION (delay less) -----
    print("Playing in fast motion...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("Fast Motion", frame)
        if cv2.waitKey(int(1000 / (fps * 2))) & 0xFF == ord('q'):  # Double speed
            break

    # Step 3: Release the video and close windows
    cap.release()
    cv2.destroyAllWindows()
