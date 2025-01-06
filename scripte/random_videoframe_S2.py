#%%
import cv2
root_path = r"C:\Users\Paula Woest\OneDrive\Desktop\Bachelore Arbeit\Paula"
# Specify the path to the video file
video_path = r"C:\Users\Paula Woest\OneDrive\Desktop\Bachelore Arbeit\Paula\2283\session2\grayscale_coh_6_2283_session2.mp4"
frame_output_path = r'C:\Users\Paula Woest\OneDrive\Desktop\Bachelore Arbeit\Paula\output_data\Session 2_Interaction\frame_output.jpg'

# Open the video file
cap = cv2.VideoCapture(video_path)

# Check if the video was successfully opened
if not cap.isOpened():
    print("Error: Cannot open video file.")
    exit()

# Specify the frame number to extract (e.g., frame 100)
frame_number = 100
cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

# Read the specified frame
ret, frame = cap.read()

if ret:
    # Save the extracted frame as an image file
    cv2.imwrite(frame_output_path, frame)
    print(f"Frame {frame_number} has been saved as {frame_output_path}.")
else:
    print(f"Error: Could not read frame {frame_number}.")

# Release the video capture object
cap.release()
