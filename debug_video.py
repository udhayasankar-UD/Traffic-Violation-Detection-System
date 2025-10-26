import cv2

# --- Configuration ---
# Make sure this path is correct!
VIDEO_PATH = 'test_video.mp4'
# --- End Configuration ---

def test_video_playback():
    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        print(f"Error: Could not open video file at '{VIDEO_PATH}'")
        print("Please check the following:")
        print("1. The file name is spelled correctly.")
        print("2. The file is in the same folder as the script.")
        return

    print("Successfully opened video file. Press 'q' to quit.")

    frame_count = 0
    while True:
        ret, frame = cap.read()
        
        # If 'ret' is False, it means the video has ended or there was an error
        if not ret:
            print("Could not read frame from video, or video has ended.")
            break

        frame_count += 1
        print(f"Displaying frame number: {frame_count}")

        # Display the frame
        cv2.imshow("Video Test", frame)

        # Wait for 25ms and check if 'q' is pressed
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()
    print("Playback finished.")

if __name__ == '__main__':
    test_video_playback()