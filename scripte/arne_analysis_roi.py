# %%
import cv2
import numpy as np
import os
from csv import writer


# %%
def click_event(event, x, y, flags, params):
    """
    Function to record x- and y-coordinates of left mouse-click
    """
    if event == cv2.EVENT_LBUTTONDOWN:
        # Draw a red circle at the clicked point
        cv2.circle(params["frame"], (x, y), 4, (0, 0, 255), -1)
        cv2.imshow("Frame", params["frame"])
        params["left"].append((x, y))
        print(x, y)

        # Check if the desired number of clicks is reached
        if len(params["left"]) >= 4:
            # Wait for 2 milliseconds
            cv2.waitKey(500)
            cv2.destroyAllWindows()

    if event == cv2.EVENT_RBUTTONDOWN:
        params["right"].append((x, y))
        print(x, y)


# %%
def display_frame(video_path, frame_number):
    """
    display random frame from video where users click cornes of box to determine ROI
    """
    # open video file#
    vid = cv2.VideoCapture(video_path)

    # set frame position
    vid.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    # read frame
    success, frame = vid.read()
    # greyFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # check frame read successful
    if success:
        # display frame
        # cv2.imshow("Frame", greyFrame)
        cv2.imshow("Frame", frame)

        # Create a dictionary to store coordinates and the frame
        # coordinates = {"frame": greyFrame, "left": [], "right": []}
        coordinates = {"frame": frame, "left": [], "right": []}

        cv2.setMouseCallback("Frame", click_event, coordinates)

        # Loop to wait for key press or window close event
        while True:
            key = cv2.waitKey(1)  # Wait for 1 millisecond
            if key & 0xFF == ord("q"):  # Check if 'q' key is pressed
                break  # Exit the loop if 'q' key is pressed
            if (
                cv2.getWindowProperty("Frame", cv2.WND_PROP_VISIBLE) < 1
            ):  # Check if window is closed
                break  # Exit the loop if window is closed
    else:
        print(f"Error reading frame {frame_number}")

    # release video capture object
    vid.release()

    # close all OpenCV windows
    cv2.destroyAllWindows()

    return coordinates


# %%
def display_rectangle_save_roi(
    video_path,
    frame_number,
    original_vertices,
    cohort,
    mouse_id,
    object,
    orientation,
    scale_factor=1.5,
):
    """
    Display the frame with original and scaled rectangles to ensure ROI is created successfully.
    """
    # Open video file
    vid = cv2.VideoCapture(video_path)

    # Set frame position
    vid.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    # Read frame
    success, frame = vid.read()
    greyframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Check frame read successful
    if success:

        # Calculate the center of the rectangle
        center = np.mean(original_vertices, axis=0)

        # Calculate scaled vertices
        scaled_vertices = center + scale_factor * (original_vertices - center)

        # Draw original rectangle
        cv2.polylines(
            greyframe,
            [original_vertices.astype(int)],
            isClosed=True,
            color=(0, 255, 0),
            thickness=2,
        )

        # Draw scaled rectangle
        cv2.polylines(
            greyframe,
            [scaled_vertices.astype(int)],
            isClosed=True,
            color=(255, 0, 0),
            thickness=2,
        )

        # Save the modified frame
        save_path = f"{cohort}_{mouse_id}_{object}_{orientation}.jpg"
        cv2.imwrite(save_path, greyframe)

        # Display frame
        cv2.imshow("Frame", greyframe)

        # Loop to wait for key press or window close event
        while True:
            key = cv2.waitKey(1)  # Wait for 1 millisecond
            if key & 0xFF == ord("q"):  # Check if 'q' key is pressed
                break  # Exit the loop if 'q' key is pressed
            if (
                cv2.getWindowProperty("Frame", cv2.WND_PROP_VISIBLE) < 1
            ):  # Check if window is closed
                break  # Exit the loop if window is closed
    else:
        print(f"Error reading frame {frame_number}")

    # Release video capture object
    vid.release()

    # Close all OpenCV windows
    cv2.destroyAllWindows()

    return scaled_vertices, center


# %%
# TODO change video path
video_path = r"C:\Users\neuba\OneDrive\Bremen_Neurosciences\Lab_Project_2\SLEAP_Workshop_AG_Masseck\SLEAP_Workshop\SLEAP_Workshop_Two_Mice.mp4"
vid = cv2.VideoCapture(video_path)
frame_number = 0

# %%
cohort = input("Enter Cohort:")
mouse_id = input("Enter Mouse ID")
orientation = input("Enter Orientation, NW or NE")

# %%
coordinates1 = display_frame(video_path, frame_number)
boundaries_object1 = np.array(coordinates1["left"], dtype=np.float64)
scaled_vertices1, center_obj1 = display_rectangle_save_roi(
    video_path, frame_number, boundaries_object1, cohort, mouse_id, "obj1", orientation
)


# %%
coordinates2 = display_frame(video_path, frame_number)
boundaries_object2 = np.array(coordinates2["left"], dtype=np.float64)
scaled_vertices2, center_obj2 = display_rectangle_save_roi(
    video_path, frame_number, boundaries_object2, cohort, mouse_id, "obj2", orientation
)

# %%
x1_obj1, y1_obj1 = scaled_vertices1[0]
x2_obj1, y2_obj1 = scaled_vertices1[1]
x3_obj1, y3_obj1 = scaled_vertices1[2]
x4_obj1, y4_obj1 = scaled_vertices1[3]
xc_obj1, yc_obj1 = center_obj1
x1_obj2, y1_obj2 = scaled_vertices2[0]
x2_obj2, y2_obj2 = scaled_vertices2[1]
x3_obj2, y3_obj2 = scaled_vertices2[2]
x4_obj2, y4_obj2 = scaled_vertices2[3]
xc_obj2, yc_obj2 = center_obj2

# %%
# List that we want to add as a new row
header = [
    "cohort",
    "mouse_id",
    "video_path",
    "orientation",
    "x1_obj1",
    "y1_obj1",
    "x2_obj1",
    "y2_obj1",
    "x3_obj1",
    "y3_obj1",
    "x4_obj1",
    "y4_obj1",
    "xc_obj1",
    "yc_obj1",
    "x1_obj2",
    "y1_obj2",
    "x2_obj2",
    "y2_obj2",
    "x3_obj2",
    "y3_obj2",
    "x4_obj2",
    "y4_obj2",
    "xc_obj2",
    "yc_obj2",
]
video_info = [
    cohort,
    mouse_id,
    video_path,
    orientation,
    x1_obj1,
    y1_obj1,
    x2_obj1,
    y2_obj1,
    x3_obj1,
    y3_obj1,
    x4_obj1,
    y4_obj1,
    xc_obj1,
    yc_obj1,
    x1_obj2,
    y1_obj2,
    x2_obj2,
    y2_obj2,
    x3_obj2,
    y3_obj2,
    x4_obj2,
    y4_obj2,
    xc_obj2,
    yc_obj2,
]

# Check if the CSV file exists
file_exists = os.path.isfile("roi_info_objects.csv")

# Open our existing CSV file in append mode
# Create a file object for this file
# nz for new zone analysis
with open("roi_info_objects.csv", "a") as f_object:  # a is for append

    # Pass this file object to csv.writer() and get a writer object
    writer_object = writer(f_object)

    # Write the header row if the file doesn't exist
    if not file_exists:
        writer_object.writerow(header)

    # Write the data row
    writer_object.writerow(video_info)

# %%
import pandas as pd

# TODO separate script
video_path = r""
csv_path = r""

vid = cv2.VideoCapture(video_path)
total_frame_count = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
fps = vid.get(cv2.CAP_PROP_FPS)

cohort = input("Enter Cohort:")
mouse_id = input("Enter Mouse ID")
orientation = input("Enter Orientation, NW or NE")

df_roi_info = pd.read_csv("roi_info_objects.csv")
df_roi_info_coh = df_roi_info[df_roi_info["orientation"] == orientation]

x1_obj1 = df_roi_info_coh.x1_obj1.values[0]
x3_obj1 = df_roi_info_coh.x3_obj1.values[0]
y4_obj1 = df_roi_info_coh.y4_obj1.values[0]
y2_obj1 = df_roi_info_coh.y2_obj1.values[0]

df_mouse = pd.read_csv(csv_path)


# %%
def empty_frame_tensor(video_path):
    vid = cv2.VideoCapture(video_path)
    mask_frames = np.zeros(
        (int((vid.get(cv2.CAP_PROP_FRAME_COUNT))), 3, 4), np.float64
    )  # 3 cause 3 skeleton nodes, 4 cause 2 dimensions x 2 boolean (True/False)
    return mask_frames


# %% create empty tensor
mask_frames_obj1 = empty_frame_tensor(video_path)

# %%
# all x-column stored in dimension 1 of tensor
columns_x_dim = ["Nose.x", "Neck.x", "Tail_Base.x"]

# all y-column stored in dimension 2 of tensor
columns_y_dim = ["Nose.y", "Neck.y", "Tail_Base.y"]

# Assign values from DataFrame to the tensor
mask_frames_obj1[:, :, 0] = df_mouse[columns_x_dim].to_numpy()
mask_frames_obj1[:, :, 1] = df_mouse[columns_y_dim].to_numpy()

# %%
x_coords_mouse = mask_frames_obj1[:, :, 0]
y_coords_mouse = mask_frames_obj1[:, :, 1]

# boolean vector whether body parts from bl6 mouse are inside ROI
x_mask_mouse_obj1 = np.logical_and(x_coords_mouse <= x3_obj1, x_coords_mouse >= x1_obj1)
y_mask_mouse_obj1 = np.logical_and(y_coords_mouse >= y4_obj1, y_coords_mouse <= y2_obj1)

# assign boolean vectors to tensor
mask_frames_obj1[:, :, 2] = x_mask_mouse_obj1
mask_frames_obj1[:, :, 3] = y_mask_mouse_obj1

# %%
# bl6 mouse in roi when both boolean vector for x and boolean vector for y coordinate True
in_roi_obj1 = np.logical_and(x_mask_mouse_obj1 == 1, y_mask_mouse_obj1 == 1)

# gives you row names/index, ergo frame numbers for nose --> nose is first body part in tensor --> in_roi[:, 0]
frame_idx_nose_in_roi_obj1 = np.where(in_roi_obj1[:, 0] == 1)[0]
# total number of frames Neck in roi
total_frames_nose_in_roi_obj1 = np.sum(in_roi_obj1[:, 0] == 1)

# create dataframe
df_expl_obj1 = pd.DataFrame(in_roi_obj1)
# rename value columns
df_expl_obj1.columns = ["Nose", "Neck", "Tail_Base"]

# rename index column
df_expl_obj1 = df_expl_obj1.rename_axis("frame_idx")

# create additional column
df_expl_obj1["frame_number"] = df_mouse["frame_idx"].values


# %% Function to calculate the angle between two vectors
def angle_between_vectors(vector1, vector2):
    """
    calculates angle between two vectors
    """
    dot_product = np.sum(vector1 * vector2, axis=1)
    magnitude_product = np.linalg.norm(vector1, axis=1) * np.linalg.norm(
        vector2, axis=1
    )
    cosine_angle = dot_product / magnitude_product
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)


# %%
def calculate_cross_product(vector1, vector2):
    """
    calculates cross products between two vectors
    """
    return np.cross(vector1, vector2)


# %%
# get x- and y- values of neck and nose of bl6 mouse
v_x_neck_mouse = df_mouse["Neck.x"].values
v_y_neck_mouse = df_mouse["Neck.y"].values
v_x_nose_mouse = df_mouse["Nose.x"].values
v_y_nose_mouse = df_mouse["Nose.y"].values

# get x- and y-value of centre object 1
xc_obj1 = df_roi_info_coh.xc_obj1.values[0]
yc_obj1 = df_roi_info_coh.yc_obj1.values[0]

# x- and y- coordinates of origin
x_origin = 0
y_origin = 0

# calculate origin vectors
o_neck_mouse = np.array([v_x_neck_mouse - x_origin, v_y_neck_mouse - y_origin]).T
o_nose_mouse = np.array([v_x_nose_mouse - x_origin, v_y_nose_mouse - y_origin]).T
o_c_obj1 = np.array([xc_obj1 - x_origin, yc_obj1 - y_origin]).T

# calculate neck-nose and nose-mid_box vectors
vector_neck_nose = o_nose_mouse - o_neck_mouse
vector_neck_c_obj1 = o_neck_mouse - o_c_obj1

# calculate the angle between the vectors
angles_obj1 = angle_between_vectors(vector_neck_nose, vector_neck_c_obj1)

# calculate the cross product for every frame
cross_products_obj1 = calculate_cross_product(vector_neck_nose, vector_neck_c_obj1)

# %%
# angle condition
cond_degree = 20  # mice have 40° field of vision --> 40°/2 = 20°
cond_angles = angles_obj1 >= cond_degree
df_expl_obj1["cond_angles"] = cond_angles

# direction condition
cond_direction = 0
cond_direction = cross_products_obj1 > cond_direction
df_expl_obj1["cond_direction"] = cond_direction

# %%
frame_idx_expl_obj1 = np.where(
    (df_expl_obj1["cond_angles"] == 1)
    & (df_expl_obj1["cond_direction"] == 1)
    & (df_expl_obj1["Nose"] == 1)
)[
    0
]  # which frames meet requirements/conditions

total_frames_expl_obj1 = np.where(
    (df_expl_obj1["cond_angles"] == 1)
    & (df_expl_obj1["cond_direction"] == 1)
    & (df_expl_obj1["Nose"] == 1)
)[0].shape[
    0
]  # number of frames for which conditions are true


# Save in_roi_ang_dir dataframe as .csv file
df_expl_obj1.to_csv("explore_obj1.csv")


# %%
s_expl_obj1 = total_frames_expl_obj1 / fps

# %%
header = ["cohort", "mouse_id", "orientation", "s_expl_obj1", "s_expl_obj2"]
seconds_info = [cohort, mouse_id, orientation, s_expl_obj1, s_expl_obj2]

# Check if the CSV file exists
file_exists = os.path.isfile("expl_seconds_info.csv")

# Open our existing CSV file in append mode
# Create a file object for this file
# nz for new zone analysis
with open("expl_seconds_info.csv", "a") as f_object:  # a is for append

    # Pass this file object to csv.writer() and get a writer object
    writer_object = writer(f_object)

    # Write the header row if the file doesn't exist
    if not file_exists:
        writer_object.writerow(header)

    # Write the data row
    writer_object.writerow(seconds_info)
