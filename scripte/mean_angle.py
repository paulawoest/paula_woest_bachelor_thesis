# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import math

# %%
# Definiere den Dateipfad
root_path = r"C:\Users\Paula Woest\OneDrive\Desktop\Bachelore Arbeit\Paula"


# %%
def load_data(root_path):
    """
    Load all cleand data into one dataframe.
    """
    data = []
    for mouse_folder_path in Path(root_path).iterdir():
        if not mouse_folder_path.is_dir():
            continue
        if not mouse_folder_path.name.isdigit():
            continue
        for session_path in mouse_folder_path.iterdir():
            for csv_path in session_path.iterdir():
                if not csv_path.name.endswith("bl6_data_cleaned.csv"):
                    continue
                print(f"Loading csv file: {csv_path.name}")
                df = pd.read_csv(csv_path, index_col=0)
                mouse_ID = int(mouse_folder_path.name)
                df.insert(0, "mouse_ID", [mouse_ID] * len(df.index))
                session = int(session_path.name[-1])
                df.insert(1, "session", [session] * len(df.index))
                data.append(df)
    return pd.concat(data, axis=0)


all_data = load_data(root_path)
# coords = all_data[["mouse_ID", "session", "Nose.x", "Nose.y", "Neck.x", "Neck.y", "Tail_Base.x", "Tail_Base.y"]]
# groups = coords.groupby(["mouse_ID", "session"])
# return data, um einzelne dataframes zu erreichen

# %%
cohort = 7
df_video_info = pd.read_csv(
    r"C:\Users\Paula Woest\OneDrive\Desktop\Bachelore Arbeit\Paula\output_data\video_info.csv"
)
df_video_info_coh = df_video_info[df_video_info["cohort"] == cohort]
# get coordinates for calculating the middle of the box
x_bound_left_cd1 = df_video_info_coh.x_bound_left_cd1.values[0]
x_bound_right_cd1 = df_video_info_coh.x_bound_right_cd1.values[0]
y_bound_bl6 = df_video_info_coh.y_bound_bl6.values[0]

midB_7 = np.zeros((2), dtype=np.float64)
midB_7[0] = (x_bound_left_cd1 + x_bound_right_cd1) / 2  # x coordinates
midB_7[1] = y_bound_bl6

midB_7[1] = midB_7[1] * (2 / 3)
# %%
cohort = 6
df_video_info = pd.read_csv(
    r"C:\Users\Paula Woest\OneDrive\Desktop\Bachelore Arbeit\Paula\output_data\video_info.csv"
)
df_video_info_coh = df_video_info[df_video_info["cohort"] == cohort]
# get coordinates for calculating the middle of the box
x_bound_left_cd1 = df_video_info_coh.x_bound_left_cd1.values[0]
x_bound_right_cd1 = df_video_info_coh.x_bound_right_cd1.values[0]
y_bound_bl6 = df_video_info_coh.y_bound_bl6.values[0]

midB_6 = np.zeros((2), dtype=np.float64)
midB_6[0] = (x_bound_left_cd1 + x_bound_right_cd1) / 2  # x coordinates
midB_6[1] = y_bound_bl6

midB_6[1] = midB_6[1] * (2 / 3)


# %%
# center
def calc_angle(u, v):
    u = np.array(u)
    v = np.array(v)

    cross_product = np.cross(u, v)
    magnitude_u = np.linalg.norm(u)
    magnitude_v = np.linalg.norm(v)
    magnitude_cross = np.linalg.norm(cross_product)

    sin_theta = magnitude_cross / (magnitude_u * magnitude_v)
    sin_theta = np.clip(
        sin_theta, -1.0, 1.0
    )  # Ensure sin_theta is within the valid range

    angle_radians = np.arcsin(sin_theta)
    angle_degrees = np.degrees(angle_radians)

    return angle_degrees


def calc_angle_center(row, midB):
    ref_x = midB[0]
    ref_y = midB[1]
    neck_x, neck_y, nose_x, nose_y = row[
        ["Neck.x", "Neck.y", "Nose.x", "Nose.y"]
    ]  # args = argumnets
    u = [(nose_x - neck_x), (nose_y - neck_y)]
    v = [(ref_x - neck_x), (ref_y - neck_y)]

    return calc_angle(u, v)


# num_frames = len(coordinates)

# Step 2: Divide by two to find the midpoint
# half_point = num_frames // 2
# %%
data1 = all_data[all_data["session"] == 1]
data6 = data1[data1["mouse_ID"] <= 2301]  # change the numbers of the different
data7 = data1[data1["mouse_ID"] > 2301]
angle_col6 = data6.apply(calc_angle_center, axis=1, args=(midB_6,))
angle_col7 = data7.apply(calc_angle_center, axis=1, args=(midB_7,))
data6.loc[:, "angle"] = angle_col6
data7.loc[:, "angle"] = angle_col7

# %%
colnames = ["angle"]
feat_names = [
    feat + "_" + name
    # fh = first half, sh = second half
    for feat in "mean".split()
    for name in colnames
]
feat_names = ["track", "mouse_ID", *feat_names]

# %%
feature_matrix_1_6 = []
for (track, mouse_ID), data in data6.groupby(["track", "mouse_ID"]):

    angle = data[["angle"]]

    # Step 1: Count the number of frames
    num_frames = len(angle)

    # Step 2: Divide by two to find the midpoint
    one_minute = num_frames // 2.53  # Integer division
    one_minute_int = round(one_minute)  # Integer division

    """
    mean_loc_first_half = angle.iloc[:half_point].mean()
    mean_loc_second_half = angle.iloc[half_point:].mean()
    median_loc_first_half = angle.iloc[:half_point].median()
    median_loc_second_half = angle.iloc[half_point:].median()
    std_loc_first_half = angle.iloc[:half_point].std()
    std_loc_second_half = angle.iloc[half_point:].std()
    """

    new_col = [
        track,
        mouse_ID,
        *angle.iloc[:one_minute_int].mean(),
        # *angle.iloc[half_point:].mean(),
    ]

    feature_matrix_1_6.append(new_col)

feature_matrix_1_6 = pd.DataFrame(feature_matrix_1_6, columns=feat_names)


# %%
feature_matrix_1_7 = []
for (track, mouse_ID), data in data7.groupby(["track", "mouse_ID"]):

    angle = data[["angle"]]

    # Step 1: Count the number of frames
    num_frames = len(angle)

    # Step 2: Divide by two to find the midpoint
    one_minute = num_frames // 2.53  # Integer division
    one_minute_int = round(one_minute)  # Integer division

    """
    mean_loc_first_half = angle.iloc[:half_point].mean()
    mean_loc_second_half = angle.iloc[half_point:].mean()
    median_loc_first_half = angle.iloc[:half_point].median()
    median_loc_second_half = angle.iloc[half_point:].median()
    std_loc_first_half = angle.iloc[:half_point].std()
    std_loc_second_half = angle.iloc[half_point:].std()
    """

    new_col = [
        track,
        mouse_ID,
        *angle.iloc[:one_minute_int].mean(),
        # *angle.iloc[half_point:].mean(),
    ]

    feature_matrix_1_7.append(new_col)

feature_matrix_1_7 = pd.DataFrame(feature_matrix_1_7, columns=feat_names)

# %%
# Concatenate along rows (add rows)
feature_matrix_1 = pd.concat(
    [feature_matrix_1_6, feature_matrix_1_7], axis=0, ignore_index=True
)

# Concatenate along columns (add columns)
# result_columns = pd.concat([matrix1, matrix2], axis=1)


# %%
def load_data(root_path):
    """
    Load all cleand data into one dataframe.
    """
    data = []
    for mouse_folder_path in Path(root_path).iterdir():
        if not mouse_folder_path.is_dir():
            continue
        if not mouse_folder_path.name.isdigit():
            continue
        for session_path in mouse_folder_path.iterdir():
            for csv_path in session_path.iterdir():
                if not csv_path.name.endswith("data_cleaned.csv"):
                    continue
                print(f"Loading csv file: {csv_path.name}")
                df = pd.read_csv(csv_path, index_col=0)
                mouse_ID = int(mouse_folder_path.name)
                df.insert(0, "mouse_ID", [mouse_ID] * len(df.index))
                session = int(session_path.name[-1])
                df.insert(1, "session", [session] * len(df.index))
                data.append(df)
    return pd.concat(data, axis=0)


all_data = load_data(root_path)
# %%
data2 = all_data[all_data["session"] == 2]
# %%
data_bl6 = data2[data2["track"] == "bl6"]
data_cd1 = data2[data2["track"] == "cd1"]

# %%
test = data_bl6.copy()
test["Neck_cd1.x"] = data_cd1["Neck.x"]
test["Neck_cd1.y"] = data_cd1["Neck.y"]


# %%
def calc_angle_row(u, v, row):
    u = np.array(u)
    v = np.array(v)

    cross_product = np.cross(u, v)
    magnitude_u = np.linalg.norm(u)
    magnitude_v = np.linalg.norm(v)
    magnitude_cross = np.linalg.norm(cross_product)

    sin_theta = magnitude_cross / (magnitude_u * magnitude_v)
    sin_theta = np.clip(sin_theta, -1.0, 1.0)  # Clip sin_theta to valid range

    angle_radians = np.arcsin(sin_theta)
    angle_degrees = np.degrees(angle_radians)

    return angle_degrees


# Define calc_angle_interaction to process each row
def calc_angle_interaction(row):
    neck_x_cd1, neck_y_cd1 = row[["Neck_cd1.x", "Neck_cd1.y"]]
    neck_x_bl6, neck_y_bl6, nose_x_bl6, nose_y_bl6 = row[
        ["Neck.x", "Neck.y", "Nose.x", "Nose.y"]
    ]
    u = [(nose_x_bl6 - neck_x_bl6), (nose_y_bl6 - neck_y_bl6)]
    v = [(neck_x_cd1 - neck_x_bl6), (neck_y_cd1 - neck_y_bl6)]

    # Pass row as the third argument
    angle = calc_angle_row(u, v, row)

    return angle


# %%
test["angle"] = test.apply(calc_angle_interaction, axis=1)

# %%
colnames = ["angle"]
feat_names = [
    feat + "_" + name
    # fh = first half, sh = second half
    for feat in "mean".split()
    for name in colnames
]
feat_names = ["track", "mouse_ID", *feat_names]

# %%
feature_matrix_2 = []
for (track, mouse_ID), data in test.groupby(["track", "mouse_ID"]):

    angle = data[["angle"]]

    # Step 1: Count the number of frames
    num_frames = len(angle)

    # Step 2: Divide by two to find the midpoint
    one_minute = num_frames // 2.53  # Integer division
    one_minute_int = round(one_minute)  # Integer division

    """
    mean_loc_first_half = angle.iloc[:half_point].mean()
    mean_loc_second_half = angle.iloc[half_point:].mean()
    median_loc_first_half = angle.iloc[:half_point].median()
    median_loc_second_half = angle.iloc[half_point:].median()
    std_loc_first_half = angle.iloc[:half_point].std()
    std_loc_second_half = angle.iloc[half_point:].std()
    """

    new_col = [
        track,
        mouse_ID,
        *angle.iloc[:one_minute_int].mean(),
        # *angle.iloc[half_point:].mean(),
    ]

    feature_matrix_2.append(new_col)

feature_matrix_2 = pd.DataFrame(feature_matrix_2, columns=feat_names)

# %%
# interaction angle betweeen tracks for mouse 2283
# Filter data for the example mouse and ensure both tracks are present
example_mouse_id = 2283

# Filter the test DataFrame for the specific mouse
example_data = test[test["mouse_ID"] == example_mouse_id]

# Ensure the data is sorted by frame index (assuming index represents time)
example_data = example_data.sort_index()

# Plot the angle over time
plt.figure(figsize=(12, 6))
plt.plot(
    example_data.index,  # Assuming index represents the frame/time
    example_data["angle"],  # Angle between bl6 and cd1
    marker="o",
    linestyle="-",
    color="blue",
    label=f"Angle Between Tracks (Mouse {example_mouse_id})",
)

# Add plot labels and title
plt.title(f"Interaction Angle Between Tracks for Mouse {example_mouse_id}", fontsize=16)
plt.xlabel("Frame Index (Time)", fontsize=14)
plt.ylabel("Angle (Degrees)", fontsize=14)

# Add a legend
plt.legend(fontsize=12)

# Add grid for better readability
plt.grid(True)

# Display the plot
plt.show()

# %%
# Filter data for mouse 2283
example_mouse_id = 2283
cd1_data = test[test["mouse_ID"] == example_mouse_id]

# Extract `cd1` neck coordinates
cd1_x = cd1_data["Neck_cd1.x"]
cd1_y = cd1_data["Neck_cd1.y"]

# Extract arena boundaries from the video_info DataFrame (adjust as needed)
arena_x_min = test["Neck.x"].min() - 10  # Add padding for better visualization
arena_x_max = test["Neck.x"].max() + 10
arena_y_min = test["Neck.y"].min() - 10
arena_y_max = test["Neck.y"].max() + 10

# Define the "little box" (adjust boundaries for your context)
little_box_x_min = cd1_x.min() - 2  # Add padding for better visualization
little_box_x_max = cd1_x.max() + 2
little_box_y_min = cd1_y.min() - 2
little_box_y_max = cd1_y.max() + 2

# Create the spatial plot
plt.figure(figsize=(10, 8))

# Plot the full arena
plt.plot(
    [arena_x_min, arena_x_max, arena_x_max, arena_x_min, arena_x_min],
    [arena_y_min, arena_y_min, arena_y_max, arena_y_max, arena_y_min],
    linestyle="--",
    color="gray",
    label="Arena Boundary",
)

# Highlight the "little box" where the `cd1` mouse is located
plt.plot(
    [
        little_box_x_min,
        little_box_x_max,
        little_box_x_max,
        little_box_x_min,
        little_box_x_min,
    ],
    [
        little_box_y_min,
        little_box_y_min,
        little_box_y_max,
        little_box_y_max,
        little_box_y_min,
    ],
    linestyle="-",
    color="red",
    label="cd1 Box Boundary",
)

# Plot the trajectory of the `cd1` neck
plt.plot(
    cd1_x, cd1_y, marker="o", linestyle="-", color="blue", label="cd1 Neck Trajectory"
)

# Highlight start and end points
plt.scatter(
    cd1_x.iloc[0],
    cd1_y.iloc[0],
    color="green",
    label="Start",
    s=100,
    edgecolors="black",
)
plt.scatter(
    cd1_x.iloc[-1],
    cd1_y.iloc[-1],
    color="orange",
    label="End",
    s=100,
    edgecolors="black",
)

# Add labels, legend, and title
plt.title(f"Spatial Plot of Arena and cd1 Neck (Mouse {example_mouse_id})", fontsize=16)
plt.xlabel("X Coordinate", fontsize=14)
plt.ylabel("Y Coordinate", fontsize=14)
plt.legend(fontsize=12)
plt.grid(True)

# Set aspect ratio for spatial accuracy
plt.gca().set_aspect("equal", adjustable="box")

# Show the plot
plt.show()

# %%
import cv2
import matplotlib.pyplot as plt
import numpy as np

# Video path and frame index
video_path = r"C:\Users\Paula Woest\OneDrive\Desktop\Bachelore Arbeit\Paula\2283\session2\grayscale_coh_6_2283_session2.mp4"
frame_index = 4500

# Load the video and extract the desired frame
cap = cv2.VideoCapture(video_path)
cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
ret, frame = cap.read()
cap.release()

if not ret:
    raise ValueError(f"Frame {frame_index} could not be read from {video_path}")
neck_x_cd1, neck_y_cd1 = row[["Neck_cd1.x", "Neck_cd1.y"]]
neck_x_bl6, neck_y_bl6, nose_x_bl6, nose_y_bl6 = row[
    ["Neck.x", "Neck.y", "Nose.x", "Nose.y"]
]
# Convert the frame from BGR (OpenCV default) to RGB (for Matplotlib)
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
# Calculate vectors
vector_bl6 = np.array([nose_x_bl6 - neck_x_bl6, nose_y_bl6 - neck_y_bl6])
vector_cd1 = np.array([neck_x_cd1 - neck_x_bl6, neck_y_cd1 - neck_y_bl6])

# Calculate angle between vectors
cos_theta = np.dot(vector_bl6, vector_cd1) / (
    np.linalg.norm(vector_bl6) * np.linalg.norm(vector_cd1)
)
angle_degrees = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))

# Create the plot
plt.figure(figsize=(10, 8))
plt.imshow(frame_rgb)  # Show the video frame as the background

# Plot vectors
plt.quiver(
    neck_x_bl6,
    neck_y_bl6,
    vector_bl6[0],
    vector_bl6[1],
    angles="xy",
    scale_units="xy",
    scale=1,
    color="#762a83",
    label="bl6 Vector (Neck to Nose)",
)
plt.quiver(
    neck_x_bl6,
    neck_y_bl6,
    vector_cd1[0],
    vector_cd1[1],
    angles="xy",
    scale_units="xy",
    scale=1,
    color="#1b7837",
    label="cd1 Vector (Neck to Neck)",
)

# Highlight key points
plt.scatter(
    [neck_x_bl6, nose_x_bl6, neck_x_cd1],
    [neck_y_bl6, nose_y_bl6, neck_y_cd1],
    color="black",
    s=20,
    zorder=5,
)
plt.text(neck_x_bl6 - 5, neck_y_bl6 + 10, "bl6 Neck", fontsize=10, color="#7fbf7b")
plt.text(nose_x_bl6 - 5, nose_y_bl6 - 10, "bl6 Nose", fontsize=10, color="#7fbf7b")
plt.text(neck_x_cd1 + 5, neck_y_cd1 - 10, "cd1 Neck", fontsize=10, color="#1b7837")

# Add the angle annotation
plt.text(
    0.95,
    0.85,
    f"Angle: {angle_degrees:.1f}°",
    fontsize=12,
    color="purple",
    bbox=dict(facecolor="white", edgecolor="purple"),
    ha="right",
    transform=plt.gca().transAxes,
)

# Remove axes (optional, for a cleaner look)
plt.axis("off")

# Show the plot
plt.legend(fontsize=10, loc="upper right")
plt.show()

# %%
import cv2
import matplotlib.pyplot as plt
import numpy as np

# Video path and frame index
video_path = r"C:\Users\Paula Woest\OneDrive\Desktop\Bachelore Arbeit\Paula\2283\session2\grayscale_coh_6_2283_session2.mp4"
frame_index = 4500

# Load the video and extract the desired frame
cap = cv2.VideoCapture(video_path)
cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
ret, frame = cap.read()
cap.release()

if not ret:
    raise ValueError(f"Frame {frame_index} could not be read from {video_path}")

# Convert the frame from BGR (OpenCV default) to RGB (for Matplotlib)
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# Calculate vectors
vector_bl6 = np.array([nose_x_bl6 - neck_x_bl6, nose_y_bl6 - neck_y_bl6])
vector_cd1 = np.array([neck_x_cd1 - neck_x_bl6, neck_y_cd1 - neck_y_bl6])

# Calculate angle between vectors
cos_theta = np.dot(vector_bl6, vector_cd1) / (
    np.linalg.norm(vector_bl6) * np.linalg.norm(vector_cd1)
)
angle_degrees = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))

# Create the plot
plt.figure(figsize=(10, 8))
plt.imshow(frame_rgb)  # Show the video frame as the background

# Plot `bl6` neck to nose vector
plt.quiver(
    neck_x_bl6,
    neck_y_bl6,
    vector_bl6[0],
    vector_bl6[1],
    angles="xy",
    scale_units="xy",
    scale=0.1,
    color="#7fbf7b",
    label="bl6 Vector (Neck to Nose)",
)

# Plot `cd1` vector from `bl6` neck
plt.quiver(
    neck_x_bl6,
    neck_y_bl6,
    vector_cd1[0],
    vector_cd1[1],
    angles="xy",
    scale_units="xy",
    scale=0.1,
    color="#1b7837",
    label="cd1 Vector (Neck to Neck)",
)

# Adjust black dots for coordinates (smaller size)
plt.scatter(
    [neck_x_bl6, nose_x_bl6, neck_x_cd1],
    [neck_y_bl6, nose_y_bl6, neck_y_cd1],
    color="black",
    s=10,
    zorder=5,
)

# Adjust label positions to avoid overlap
plt.text(neck_x_bl6 + 15, neck_y_bl6 - 10, "bl6 Neck", fontsize=12, color="#7fbf7b")
plt.text(nose_x_bl6 - 140, nose_y_bl6 - 1.5, "bl6 Nose", fontsize=12, color="#7fbf7b")
plt.text(neck_x_cd1 + 10, neck_y_cd1 - 1.2, "cd1 Neck", fontsize=12, color="#1b7837")

# Add the angle annotation above the legend with proper spacing
plt.text(
    0.92,
    0.91,
    f"Angle: {angle_degrees:.1f}°",
    fontsize=12,
    color="purple",
    bbox=dict(facecolor="white", edgecolor="purple"),
    ha="right",
    transform=plt.gca().transAxes,
)

# Add labels, legend, and title
plt.title(
    f"Angle Between Vectors for Mouse {example_mouse_id} (Frame {frame_index})",
    fontsize=16,
)

# Remove axes for a clean look (optional)
plt.axis("off")

# Show the plot
plt.legend(fontsize=12, loc="upper right", bbox_to_anchor=(0.95, 0.9))
plt.show()

# %%
import cv2
import matplotlib.pyplot as plt
import numpy as np

# Video path and frame index
video_path = r"C:\Users\Paula Woest\OneDrive\Desktop\Bachelore Arbeit\Paula\2283\session2\grayscale_coh_6_2283_session2.mp4"
frame_index = 4500

# Load the video and extract the desired frame
cap = cv2.VideoCapture(video_path)
cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
ret, frame = cap.read()
cap.release()

if not ret:
    raise ValueError(f"Frame {frame_index} could not be read from {video_path}")

# Convert the frame from BGR (OpenCV default) to RGB (for Matplotlib)
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


# Calculate vectors
vector_bl6 = np.array([nose_x_bl6 - neck_x_bl6, nose_y_bl6 - neck_y_bl6])
vector_cd1 = np.array([neck_x_cd1 - neck_x_bl6, neck_y_cd1 - neck_y_bl6])

# Calculate angle between vectors
cos_theta = np.dot(vector_bl6, vector_cd1) / (
    np.linalg.norm(vector_bl6) * np.linalg.norm(vector_cd1)
)
angle_degrees = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))

# Create the plot
plt.figure(figsize=(10, 8))
plt.imshow(frame_rgb)  # Show the video frame as the background

# Plot `bl6` neck to nose vector
plt.quiver(
    neck_x_bl6,
    neck_y_bl6,
    vector_bl6[0],
    vector_bl6[1],
    angles="xy",
    scale_units="xy",
    scale=0.1,
    color="#7fbf7b",
    label="bl6 Vector (Neck to Nose)",
)

# Plot `cd1` vector from `bl6` neck
plt.quiver(
    neck_x_bl6,
    neck_y_bl6,
    vector_cd1[0],
    vector_cd1[1],
    angles="xy",
    scale_units="xy",
    scale=0.1,
    color="#1b7837",
    label="cd1 Vector (Neck to Neck)",
)

# Adjust black dots for coordinates (smaller size)
plt.scatter(
    [neck_x_bl6, nose_x_bl6, neck_x_cd1],
    [neck_y_bl6, nose_y_bl6, neck_y_cd1],
    color="black",
    s=10,
    zorder=5,
)

# Adjust label positions to avoid overlap
plt.text(neck_x_bl6 + 15, neck_y_bl6 - 10, "bl6 Neck", fontsize=12, color="#7fbf7b")
plt.text(nose_x_bl6 - 140, nose_y_bl6 - 1.5, "bl6 Nose", fontsize=12, color="#7fbf7b")
plt.text(neck_x_cd1 + 10, neck_y_cd1 - 1.2, "cd1 Neck", fontsize=12, color="#1b7837")

# Add the angle annotation above the legend with proper spacing
plt.text(
    0.92,
    0.91,
    f"Angle: {angle_degrees:.1f}°",
    fontsize=12,
    color="purple",
    bbox=dict(facecolor="white", edgecolor="purple"),
    ha="right",
    transform=plt.gca().transAxes,
)

# Add labels, legend, and title
plt.title(
    f"Angle Between Vectors for Mouse {example_mouse_id} (Frame {frame_index})",
    fontsize=16,
)

# Remove axes for a clean look (optional)
plt.axis("off")

# Show the plot
plt.legend(fontsize=12, loc="upper right", bbox_to_anchor=(0.95, 0.9))
plt.show()

# %%
import cv2
import matplotlib.pyplot as plt
import numpy as np

# Video path and frame index
video_path = r"C:\Users\Paula Woest\OneDrive\Desktop\Bachelore Arbeit\Paula\2283\session2\grayscale_coh_6_2283_session2.mp4"
frame_index = 4500
mouse_id = 2283

# Extract data for the specific mouse and frame from the dataset
example_frame = test[(test["mouse_ID"] == mouse_id) & (test.index == frame_index)].iloc[
    0
]

# Extract coordinates for `bl6` and `cd1`
neck_x_bl6, neck_y_bl6 = example_frame["Neck.x"], example_frame["Neck.y"]
nose_x_bl6, nose_y_bl6 = example_frame["Nose.x"], example_frame["Nose.y"]
neck_x_cd1, neck_y_cd1 = example_frame["Neck_cd1.x"], example_frame["Neck_cd1.y"]

# Load the video and extract the desired frame
cap = cv2.VideoCapture(video_path)
cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
ret, frame = cap.read()
cap.release()

if not ret:
    raise ValueError(f"Frame {frame_index} could not be read from {video_path}")

# Convert the frame from BGR (OpenCV default) to RGB (for Matplotlib)
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# Calculate vectors
vector_bl6 = np.array([nose_x_bl6 - neck_x_bl6, nose_y_bl6 - neck_y_bl6])
vector_cd1 = np.array([neck_x_cd1 - neck_x_bl6, neck_y_cd1 - neck_y_bl6])

# Calculate angle between vectors
cos_theta = np.dot(vector_bl6, vector_cd1) / (
    np.linalg.norm(vector_bl6) * np.linalg.norm(vector_cd1)
)
angle_degrees = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))

# Create the plot
plt.figure(figsize=(10, 8))
plt.imshow(frame_rgb)  # Show the video frame as the background

# Plot `bl6` neck to nose vector
plt.quiver(
    neck_x_bl6,
    neck_y_bl6,
    vector_bl6[0],
    vector_bl6[1],
    angles="xy",
    scale_units="xy",
    scale=0.1,
    color="#7fbf7b",
    label="bl6 Vector (Neck to Nose)",
)

# Plot `cd1` vector from `bl6` neck
plt.quiver(
    neck_x_bl6,
    neck_y_bl6,
    vector_cd1[0],
    vector_cd1[1],
    angles="xy",
    scale_units="xy",
    scale=1,
    color="#1b7837",
    label="cd1 Vector (Neck to Neck)",
)

# Adjust black dots for coordinates (smaller size)
plt.scatter(
    [neck_x_bl6, nose_x_bl6, neck_x_cd1],
    [neck_y_bl6, nose_y_bl6, neck_y_cd1],
    color="black",
    s=10,
    zorder=5,
)

# Adjust label positions to avoid overlap
plt.text(neck_x_bl6 + 15, neck_y_bl6 + 5, "bl6 Neck", fontsize=12, color="black")
plt.text(nose_x_bl6 - 140, nose_y_bl6 + 10, "bl6 Nose", fontsize=12, color="black")
plt.text(neck_x_cd1 + 10, neck_y_cd1 - 1.2, "cd1 Neck", fontsize=12, color="black")

# Add the angle annotation above the legend with proper spacing
plt.text(
    0.92,
    0.91,
    f"Angle: {angle_degrees:.1f}°",
    fontsize=12,
    color="purple",
    bbox=dict(facecolor="white", edgecolor="purple"),
    ha="right",
    transform=plt.gca().transAxes,
)

# Add labels, legend, and title
plt.title(
    f"Angle Between Vectors for Mouse {mouse_id} (Frame {frame_index})", fontsize=16
)

# Remove axes for a clean look (optional)
plt.axis("off")

# Show the plot
plt.legend(fontsize=12, loc="upper right", bbox_to_anchor=(0.999, 0.9))
plt.show()

# %
# %%
import cv2
import matplotlib.pyplot as plt
import numpy as np

# Video path and frame index for session 1
video_path = r"C:\Users\Paula Woest\OneDrive\Desktop\Bachelore Arbeit\Paula\2283\session1\grayscale_coh_6_2283_session1.mp4"
frame_index = 4500
mouse_id = 2283

# Extract data for the specific mouse and frame from the dataset
example_frame = test[(test["mouse_ID"] == mouse_id) & (test.index == frame_index)].iloc[
    0
]

# Extract coordinates for `bl6`
neck_x_bl6, neck_y_bl6 = example_frame["Neck.x"], example_frame["Neck.y"]
nose_x_bl6, nose_y_bl6 = example_frame["Nose.x"], example_frame["Nose.y"]

# Midpoint of the box (use the arena boundaries)
arena_x_min, arena_x_max = example_frame["Arena_x_min"], example_frame["Arena_x_max"]
arena_y_min, arena_y_max = example_frame["Arena_y_min"], example_frame["Arena_y_max"]
mid_x, mid_y = (arena_x_min + arena_x_max) / 2, (arena_y_min + arena_y_max) / 2

# Load the video and extract the desired frame
cap = cv2.VideoCapture(video_path)
cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
ret, frame = cap.read()
cap.release()

if not ret:
    raise ValueError(f"Frame {frame_index} could not be read from {video_path}")

# Convert the frame from BGR (OpenCV default) to RGB (for Matplotlib)
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# Calculate vectors
vector_bl6_nose = np.array([nose_x_bl6 - neck_x_bl6, nose_y_bl6 - neck_y_bl6])
vector_bl6_midbox = np.array([mid_x - neck_x_bl6, mid_y - neck_y_bl6])

# Calculate angle between vectors
cos_theta = np.dot(vector_bl6_nose, vector_bl6_midbox) / (
    np.linalg.norm(vector_bl6_nose) * np.linalg.norm(vector_bl6_midbox)
)
angle_degrees = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))

# Create the plot
plt.figure(figsize=(10, 8))
plt.imshow(frame_rgb)  # Show the video frame as the background

# Plot `bl6` neck to nose vector
plt.quiver(
    neck_x_bl6,
    neck_y_bl6,
    vector_bl6_nose[0],
    vector_bl6_nose[1],
    angles="xy",
    scale_units="xy",
    scale=0.1,
    color="#762a83",
    label="bl6 Vector (Neck to Nose)",
)

# Plot `bl6` neck to box midpoint vector
plt.quiver(
    neck_x_bl6,
    neck_y_bl6,
    vector_bl6_midbox[0],
    vector_bl6_midbox[1],
    angles="xy",
    scale_units="xy",
    scale=0.1,
    color="#40004b",
    label="bl6 Vector (Neck to Box Midpoint)",
)

# Adjust black dots for coordinates (smaller size)
plt.scatter(
    [neck_x_bl6, nose_x_bl6, mid_x],
    [neck_y_bl6, nose_y_bl6, mid_y],
    color="black",
    s=10,
    zorder=5,
)

# Adjust label positions to avoid overlap
plt.text(neck_x_bl6 + 15, neck_y_bl6 + 5, "bl6 Neck", fontsize=12, color="#762a83")
plt.text(nose_x_bl6 - 50, nose_y_bl6 + 10, "bl6 Nose", fontsize=12, color="#762a83")
plt.text(mid_x + 15, mid_y + 5, "Box Midpoint", fontsize=12, color="#40004b")

# Add the angle annotation above the legend with proper spacing
plt.text(
    0.92,
    0.91,
    f"Angle: {angle_degrees:.1f}°",
    fontsize=12,
    color="purple",
    bbox=dict(facecolor="white", edgecolor="purple"),
    ha="right",
    transform=plt.gca().transAxes,
)

# Add labels, legend, and title
plt.title(
    f"Angle Between Vectors for Mouse {mouse_id} (Frame {frame_index}, Session 1)",
    fontsize=16,
)

# Remove axes for a clean look (optional)
plt.axis("off")

# Show the plot
plt.legend(fontsize=12, loc="upper right", bbox_to_anchor=(0.999, 0.9))
plt.show()

# %%
import cv2
import matplotlib.pyplot as plt
import numpy as np

# Video path and frame index for session 1
video_path = r"C:\Users\Paula Woest\OneDrive\Desktop\Bachelore Arbeit\Paula\2283\session1\grayscale_coh_6_2283_session1.mp4"
frame_index = 4500
mouse_id = 2283

# Extract data for the specific mouse and frame from the dataset
example_frame = test[(test["mouse_ID"] == mouse_id) & (test.index == frame_index)].iloc[
    0
]

# Extract coordinates for `bl6`
neck_x_bl6, neck_y_bl6 = example_frame["Neck.x"], example_frame["Neck.y"]
nose_x_bl6, nose_y_bl6 = example_frame["Nose.x"], example_frame["Nose.y"]

# Midpoint of the box (using cohort 6's bounding box data)
mid_x, mid_y = example_frame["box_mid_x"], example_frame["box_mid_y"]

# Load the video and extract the desired frame
cap = cv2.VideoCapture(video_path)
cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
ret, frame = cap.read()
cap.release()

if not ret:
    raise ValueError(f"Frame {frame_index} could not be read from {video_path}")

# Convert the frame from BGR (OpenCV default) to RGB (for Matplotlib)
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# Calculate vectors
vector_bl6_nose = np.array([nose_x_bl6 - neck_x_bl6, nose_y_bl6 - neck_y_bl6])
vector_bl6_midbox = np.array([mid_x - neck_x_bl6, mid_y - neck_y_bl6])

# Calculate angle between vectors
cos_theta = np.dot(vector_bl6_nose, vector_bl6_midbox) / (
    np.linalg.norm(vector_bl6_nose) * np.linalg.norm(vector_bl6_midbox)
)
angle_degrees = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))

# Create the plot
plt.figure(figsize=(10, 8))
plt.imshow(frame_rgb)  # Show the video frame as the background

# Plot `bl6` neck to nose vector
plt.quiver(
    neck_x_bl6,
    neck_y_bl6,
    vector_bl6_nose[0],
    vector_bl6_nose[1],
    angles="xy",
    scale_units="xy",
    scale=0.1,
    color="#af8dc3",
    label="bl6 Vector (Neck to Nose)",
)

# Plot `bl6` neck to box midpoint vector
plt.quiver(
    neck_x_bl6,
    neck_y_bl6,
    vector_bl6_midbox[0],
    vector_bl6_midbox[1],
    angles="xy",
    scale_units="xy",
    scale=0.1,
    color="#762a83",
    label="bl6 Vector (Neck to Box Midpoint)",
)

# Adjust black dots for coordinates (smaller size)
plt.scatter(
    [neck_x_bl6, nose_x_bl6, mid_x],
    [neck_y_bl6, nose_y_bl6, mid_y],
    color="black",
    s=10,
    zorder=5,
)

# Adjust label positions to avoid overlap
plt.text(neck_x_bl6 + 15, neck_y_bl6 + 5, "bl6 Neck", fontsize=12, color="black")
plt.text(nose_x_bl6 - 50, nose_y_bl6 + 10, "bl6 Nose", fontsize=12, color="black")
plt.text(mid_x + 15, mid_y + 5, "Box Midpoint", fontsize=12, color="black")

# Add the angle annotation above the legend with proper spacing
plt.text(
    0.92,
    0.91,
    f"Angle: {angle_degrees:.1f}°",
    fontsize=12,
    color="purple",
    bbox=dict(facecolor="white", edgecolor="purple"),
    ha="right",
    transform=plt.gca().transAxes,
)

# Add labels, legend, and title
plt.title(
    f"Angle Between Vectors for Mouse {mouse_id} (Frame {frame_index}, Session 1)",
    fontsize=16,
)

# Remove axes for a clean look (optional)
plt.axis("off")

# Show the plot
plt.legend(fontsize=12, loc="upper right", bbox_to_anchor=(0.999, 0.9))
plt.show()
# %%
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Video path and frame index for session 1
video_path = r"C:\Users\Paula Woest\OneDrive\Desktop\Bachelore Arbeit\Paula\2283\session1\grayscale_coh_6_2283_session1.mp4"
frame_index = 300
mouse_id = 2283

# Load video info and calculate the middle of the box for cohort 6
cohort = 6
df_video_info = pd.read_csv(
    r"C:\Users\Paula Woest\OneDrive\Desktop\Bachelore Arbeit\Paula\output_data\video_info.csv"
)
df_video_info_coh = df_video_info[df_video_info["cohort"] == cohort]

# Extract box boundary data for cohort 6
x_bound_left_cd1 = df_video_info_coh.x_bound_left_cd1.values[0]
x_bound_right_cd1 = df_video_info_coh.x_bound_right_cd1.values[0]
y_bound_bl6 = df_video_info_coh.y_bound_bl6.values[0]

# Calculate the midpoint of the box
midB_6 = np.zeros((2), dtype=np.float64)
midB_6[0] = (x_bound_left_cd1 + x_bound_right_cd1) / 2  # x-coordinate
midB_6[1] = y_bound_bl6 * (2 / 3)  # y-coordinate

# Extract data for the specific mouse and frame from the dataset
example_frame = data6[
    (data6["mouse_ID"] == mouse_id) & (data6.index == frame_index)
].iloc[0]

# Extract coordinates for `bl6`
neck_x_bl6, neck_y_bl6 = example_frame["Neck.x"], example_frame["Neck.y"]
nose_x_bl6, nose_y_bl6 = example_frame["Nose.x"], example_frame["Nose.y"]

# Midpoint of the box
mid_x, mid_y = midB_6[0], midB_6[1]

# Load the video and extract the desired frame
cap = cv2.VideoCapture(video_path)
cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
ret, frame = cap.read()
cap.release()

if not ret:
    raise ValueError(f"Frame {frame_index} could not be read from {video_path}")

# Convert the frame from BGR (OpenCV default) to RGB (for Matplotlib)
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# Calculate vectors
vector_bl6_nose = np.array([nose_x_bl6 - neck_x_bl6, nose_y_bl6 - neck_y_bl6])
vector_bl6_midbox = np.array([mid_x - neck_x_bl6, mid_y - neck_y_bl6])

# Calculate angle between vectors
cos_theta = np.dot(vector_bl6_nose, vector_bl6_midbox) / (
    np.linalg.norm(vector_bl6_nose) * np.linalg.norm(vector_bl6_midbox)
)
angle_degrees = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))

# Create the plot
plt.figure(figsize=(10, 8))
plt.imshow(frame_rgb)  # Show the video frame as the background

# Plot `bl6` neck to nose vector
plt.quiver(
    neck_x_bl6,
    neck_y_bl6,
    vector_bl6_nose[0],
    vector_bl6_nose[1],
    angles="xy",
    scale_units="xy",
    scale=0.1,
    color="#af8dc3",
    label="bl6 Vector (Neck to Nose)",
)

# Plot `bl6` neck to box midpoint vector
plt.quiver(
    neck_x_bl6,
    neck_y_bl6,
    vector_bl6_midbox[0],
    vector_bl6_midbox[1],
    angles="xy",
    scale_units="xy",
    scale=0.1,
    color="#762a83",
    label="bl6 Vector (Neck to Box Midpoint)",
)

# Adjust black dots for coordinates (smaller size)
plt.scatter(
    [neck_x_bl6, nose_x_bl6, mid_x],
    [neck_y_bl6, nose_y_bl6, mid_y],
    color="black",
    s=10,
    zorder=5,
)

# Adjust label positions to avoid overlap
plt.text(neck_x_bl6 + 15, neck_y_bl6 + 5, "bl6 Neck", fontsize=12, color="black")
plt.text(nose_x_bl6 - 50, nose_y_bl6 + 10, "bl6 Nose", fontsize=12, color="black")
plt.text(mid_x + 15, mid_y + 5, "Box Midpoint", fontsize=12, color="black")

# Add the angle annotation above the legend with proper spacing
plt.text(
    0.92,
    0.91,
    f"Angle: {angle_degrees:.1f}°",
    fontsize=12,
    color="purple",
    bbox=dict(facecolor="white", edgecolor="purple"),
    ha="right",
    transform=plt.gca().transAxes,
)

# Add labels, legend, and title
plt.title(
    f"Angle Between Vectors for Mouse {mouse_id} (Frame {frame_index}, Session 1)",
    fontsize=16,
)

# Remove axes for a clean look (optional)
plt.axis("off")

# Show the plot
plt.legend(fontsize=12, loc="upper right", bbox_to_anchor=(0.999, 0.9))
plt.show()
# %%
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Video path and frame index for session 1
video_path = r"C:\Users\Paula Woest\OneDrive\Desktop\Bachelore Arbeit\Paula\2283\session1\grayscale_coh_6_2283_session1.mp4"
frame_index = 500
mouse_id = 2283

# Ensure the data is only for session 1 and mouse 2283
session = 1
data6_session = data6[(data6["mouse_ID"] == mouse_id) & (data6["session"] == session)]

# Extract data for the specific frame
if frame_index not in data6_session.index:
    raise ValueError(
        f"Frame {frame_index} not found for mouse {mouse_id} in session {session}"
    )

example_frame = data6_session.loc[frame_index]

# Extract coordinates for `bl6`
neck_x_bl6, neck_y_bl6 = example_frame["Neck.x"], example_frame["Neck.y"]
nose_x_bl6, nose_y_bl6 = example_frame["Nose.x"], example_frame["Nose.y"]

# Load video info and calculate the middle of the box for cohort 6
cohort = 6
df_video_info = pd.read_csv(
    r"C:\Users\Paula Woest\OneDrive\Desktop\Bachelore Arbeit\Paula\output_data\video_info.csv"
)
df_video_info_coh = df_video_info[df_video_info["cohort"] == cohort]

# Extract box boundary data for cohort 6
x_bound_left_cd1 = df_video_info_coh.x_bound_left_cd1.values[0]
x_bound_right_cd1 = df_video_info_coh.x_bound_right_cd1.values[0]
y_bound_bl6 = df_video_info_coh.y_bound_bl6.values[0]

# Calculate the midpoint of the box
midB_6 = np.zeros((2), dtype=np.float64)
midB_6[0] = (x_bound_left_cd1 + x_bound_right_cd1) / 2  # x-coordinate
midB_6[1] = y_bound_bl6 * (2 / 3)  # y-coordinate

mid_x, mid_y = midB_6[0], midB_6[1]

# Load the video and extract the desired frame
cap = cv2.VideoCapture(video_path)
cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
ret, frame = cap.read()
cap.release()

if not ret:
    raise ValueError(f"Frame {frame_index} could not be read from {video_path}")

# Convert the frame from BGR (OpenCV default) to RGB (for Matplotlib)
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# Calculate vectors
vector_bl6_nose = np.array([nose_x_bl6 - neck_x_bl6, nose_y_bl6 - neck_y_bl6])
vector_bl6_midbox = np.array([mid_x - neck_x_bl6, mid_y - neck_y_bl6])

# Calculate angle between vectors
cos_theta = np.dot(vector_bl6_nose, vector_bl6_midbox) / (
    np.linalg.norm(vector_bl6_nose) * np.linalg.norm(vector_bl6_midbox)
)
angle_degrees = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))

# Create the plot
plt.figure(figsize=(10, 8))
plt.imshow(frame_rgb)  # Show the video frame as the background

# Plot `bl6` neck to nose vector
plt.quiver(
    neck_x_bl6,
    neck_y_bl6,
    vector_bl6_nose[0],
    vector_bl6_nose[1],
    angles="xy",
    scale_units="xy",
    scale=0.1,
    color="#af8dc3",
    label="bl6 Vector (Neck to Nose)",
)

# Plot `bl6` neck to box midpoint vector
plt.quiver(
    neck_x_bl6,
    neck_y_bl6,
    vector_bl6_midbox[0],
    vector_bl6_midbox[1],
    angles="xy",
    scale_units="xy",
    scale=1,
    color="#762a83",
    label="bl6 Vector (Neck to Box Midpoint)",
)

# Adjust black dots for coordinates (smaller size)
plt.scatter(
    [neck_x_bl6, nose_x_bl6, mid_x],
    [neck_y_bl6, nose_y_bl6, mid_y],
    color="black",
    s=10,
    zorder=5,
)

# Adjust label positions to avoid overlap
plt.text(neck_x_bl6 + 15, neck_y_bl6 + 10, "bl6 Neck", fontsize=12, color="black")
plt.text(nose_x_bl6 - 130, nose_y_bl6 + 10, "bl6 Nose", fontsize=12, color="black")
plt.text(mid_x + 15, mid_y + 5, "Box Midpoint", fontsize=12, color="black")

# Add the angle annotation above the legend with proper spacing
plt.text(
    0.92,
    0.18,
    f"Angle: {angle_degrees:.1f}°",
    fontsize=12,
    color="purple",
    bbox=dict(facecolor="white", edgecolor="purple"),
    ha="right",
    transform=plt.gca().transAxes,
)

# Add labels, legend, and title
plt.title(
    f"Angle Between Vectors for Mouse {mouse_id} (Frame {frame_index}, Session {session})",
    fontsize=16,
)

# Remove axes for a clean look (optional)
plt.axis("off")

# Show the plot
plt.legend(fontsize=12, loc="upper right", bbox_to_anchor=(0.99, 0.15))
plt.show()
# %%
# Ensure the data is only for session 1 and mouse 2283
session = 1
data6_session = data6[(data6["mouse_ID"] == mouse_id) & (data6["session"] == session)]

# Debugging outputs
print(f"Filtered data for mouse {mouse_id}, session {session}:")
print(data6.head())
print(f"Available frame indices: {data6_session.index}")


# %%
