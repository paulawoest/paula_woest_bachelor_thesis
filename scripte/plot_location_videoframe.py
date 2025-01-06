# %%
import cv2
import pandas as pd
import matplotlib.pyplot as plt


# Function to overlay coordinates on a video frame
def overlay_coords_on_frame(video_path, csv_path, target_frame):
    # Load the CSV file containing coordinates and frame numbers
    data = pd.read_csv(
        r"C:\Users\Paula Woest\OneDrive\Desktop\Bachelore Arbeit\Paula\2283\session2\SB_2283_2_uSoIn_PW_2024-11-01_bl6_data_cleaned.csv"
    )

    # Filter data for the specified target frame
    frame_data = data[data["frame_idx"] == target_frame]
    if frame_data.empty:
        print(f"No coordinate data found for frame {target_frame}.")
        return

    # Open the video file
    cap = cv2.VideoCapture(
        r"C:\Users\Paula Woest\OneDrive\Desktop\Bachelore Arbeit\Paula\2283\session2\grayscale_coh_6_2283_session2.mp4"
    )

    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    # Set the video to the target frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
    success, frame = cap.read()

    if not success:
        print(f"Error: Could not read frame {target_frame}.")
        cap.release()
        return

    # Convert the frame to RGB for Matplotlib
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Plot the frame with Matplotlib
    plt.figure(figsize=(10, 8))
    plt.imshow(frame_rgb)
    plt.axis("off")

    # Overlay coordinates on the frame
    for _, row in frame_data.iterrows():
        x, y = int(row["Neck.x"]), int(row["Neck.y"])
        plt.plot(x, y, marker="o", color="#7fbf7b", markersize=4)
        plt.text(x + 30, y - 20, f"({x}, {y})", color="blue", fontsize=10)

    plt.title(f"Frame Example 3000, Neck Coordinates of Mouse 2283, Session 2")
    plt.show()

    # Release the video capture object
    cap.release()


# Paths to the video and CSV file
video_path = r"C:\Users\Paula Woest\OneDrive\Desktop\Bachelore Arbeit\Paula\2283\session2\grayscale_coh_6_2283_session2.mp4"  # Replace with your video file path
csv_path = r"C:\Users\Paula Woest\OneDrive\Desktop\Bachelore Arbeit\Paula\2283\session2\SB_2283_2_uSoIn_PW_2024-11-01_bl6_data_cleaned.csv"  # Replace with your CSV file path

# Define the frame number you want to analyze
target_frame = 3000  # Replace with the desired frame number

# Call the function
overlay_coords_on_frame(video_path, csv_path, target_frame)

# %%
