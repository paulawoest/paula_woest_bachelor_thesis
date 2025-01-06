# %% importing libraries
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import scipy.interpolate
import statistics
from csv import writer
import tkinter as tk
from datetime import datetime
from tkinter import messagebox
from PIL import Image, ImageTk

# %%
# directory where this script/file is saved
script_dir = os.path.dirname(
    os.path.abspath(r"C:\Users\Paula Woest\OneDrive\Desktop\Bachelore Arbeit\Paula")
)
# Set the current working directory to the script directory
os.chdir(script_dir)

# %% import analysis functions
import td_id_analysis_sleap_svenja_functions as sleapf


# %%
def get_experiment_info():
    """
    Pop-up window for user-specified experimental information

    Returns:
        experimenter_name
        mouse_id
        session
        experiment_id
        analyst_name
        analysis_date
    """
    experiment_info = {}

    def store_input():
        mouse_id = entry_widgets[0].get()
        analysis_date = datetime.now().strftime("%Y-%m-%d")

        # Store the input in the experiment_info dictionary
        experiment_info.update(
            {
                "Mouse ID": mouse_id,
                "Analysis Date": analysis_date,
            }
        )

        root.destroy()

    def is_initials(name):
        return len(name) == 2 and name.isalpha()

    def is_valid_mouse_id(mouse_id):
        return mouse_id.isdigit() and len(mouse_id) == 4

    # Create the main window
    root = tk.Tk()
    root.title("Enter Experiment Information")

    # Set the size of the window
    root.geometry("300x250")

    # Create labels and entry widgets for each input
    labels = ["Mouse ID:"]
    entry_widgets = []

    for label_text in labels:
        label = tk.Label(root, text=label_text)
        label.pack()
        entry_widget = tk.Entry(root)
        entry_widget.pack()
        entry_widgets.append(entry_widget)

    # Create a button to store the input
    button = tk.Button(root, text="Store Input", command=store_input)
    button.pack()

    # Run the tkinter event loop
    root.mainloop()

    # Return the experiment_info dictionary
    return experiment_info if experiment_info else None


# %% for quicker things
experiment_info = get_experiment_info()
if experiment_info:
    for key, value in experiment_info.items():
        print(f"{key}: {value}")
else:
    print("Experiment information was not provided.")

experimenter_name = "SB"
mouse_id = 2283
session = 2
experiment_id = "uSoIn"
analyst_name = "pW"
# TODO change cohort
cohort = "6"
analysis_date = experiment_info["Analysis Date"]

# %% get experiment info
# experiment_info = sleapf.get_experiment_info()
# if experiment_info:
#     for key, value in experiment_info.items():
#         print(f"{key}: {value}")
# else:
#     print("Experiment information was not provided.")

# experimenter_name = experiment_info["Experimenter Name"]
# mouse_id = experiment_info["Mouse ID"]
# session = experiment_info["Session"]
# experiment_id = experiment_info["Experiment ID"]
# analyst_name = experiment_info["Analyst Name"]
# cohort = experiment_info["Cohort"]
# analysis_date = experiment_info["Analysis Date"]


# %% set folder path where files etc. get saved
folder_path = os.path.join(os.getcwd(), str(mouse_id), f"session{session}")

# %% select file paths
video_path = os.path.join(folder_path, f"{cohort}_{mouse_id}_session{session}.mp4")
# %%
vid = cv2.VideoCapture(video_path)
total_frame_count = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
fps = vid.get(cv2.CAP_PROP_FPS)

# %%
df_video_info = pd.read_csv(r"D:\nz_td_id_video_info.csv")
df_video_info_coh = df_video_info[df_video_info["cohort"] == cohort]


# %% set boundaries for cd1 mouse, i.e., cd1 mouse cannot be outside of box
x_bound_left_cd1 = df_video_info_coh.x_bound_left_cd1.values[0]
x_bound_right_cd1 = df_video_info_coh.x_bound_right_cd1.values[0]
y_bound_cd1 = df_video_info_coh.y_bound_cd1.values[0]

# %% set boundaries for bl6 mouse, i.e., bl6 mouse cannot be inside of box
x_bound_left_bl6 = df_video_info_coh.x_bound_left_bl6.values[0]
x_bound_right_bl6 = df_video_info_coh.x_bound_right_bl6.values[0]
y_bound_bl6 = df_video_info_coh.y_bound_bl6.values[0]

# %% get boundaries of ROI
xleftT = df_video_info_coh.xleftT.values[0]
yleftT = df_video_info_coh.yleftT.values[0]
xrightB = df_video_info_coh.xrightB.values[0]
yrightB = df_video_info_coh.yrightB.values[0]

# %% read cleaned tracks in again
# TODO: change date here in case it's not 2024-04-10
csv_path_cleaned = os.path.join(
    folder_path,
    f"{experimenter_name}_{mouse_id}_{session}_{experiment_id}_{analyst_name}_2024-04-10_tracks_switched.csv",
)
# split dataframe according to mouse
df_bl6, df_cd1 = sleapf.split_data(csv_path_cleaned)
df_bl6 = df_bl6.reset_index(drop=True)
df_cd1 = df_cd1.reset_index(drop=True)

# create new normalised column (range 0-1), scaled by min and max possible values
df_bl6 = sleapf.scale_column(df_bl6, "Neck.score", 0, 1)
df_cd1 = sleapf.scale_column(df_cd1, "Neck.score", 0, 1)

# name attribute for plot
df_bl6.name = "bl6"
df_cd1.name = "cd1"

# %% show histogram of Neck.score and Neck.score scaled for bl6
sleapf.plot_histogram_with_percentile(
    df_bl6,
    "Neck.score",
    folder_path,
    "bl6_neck_score",
    experimenter_name,
    mouse_id,
    session,
    experiment_id,
    analyst_name,
    analysis_date,
    percentile=0.05,
    num_bins=20,
)
sleapf.plot_histogram_with_percentile(
    df_bl6,
    "Neck.score_scaled",
    folder_path,
    "bl6_neck_score_scaled",
    experimenter_name,
    mouse_id,
    session,
    experiment_id,
    analyst_name,
    analysis_date,
    percentile=0.05,
    num_bins=20,
)

# %% show histogram of Neck.score and Neck.score scaled for cd1
sleapf.plot_histogram_with_percentile(
    df_cd1,
    "Neck.score",
    folder_path,
    "cd1_neck_score",
    experimenter_name,
    mouse_id,
    session,
    experiment_id,
    analyst_name,
    analysis_date,
    percentile=0.05,
    num_bins=20,
)
sleapf.plot_histogram_with_percentile(
    df_cd1,
    "Neck.score_scaled",
    folder_path,
    "cd1_neck_score_scaled",
    experimenter_name,
    mouse_id,
    session,
    experiment_id,
    analyst_name,
    analysis_date,
    percentile=0.05,
    num_bins=20,
)

# %% replace zeros in 'Neck.score' column with nan
df_bl6["Neck.score"] = df_bl6["Neck.score"].replace(0, np.nan)
df_bl6["Neck.score_scaled"] = df_bl6["Neck.score_scaled"].replace(0, np.nan)

df_cd1["Neck.score"] = df_cd1["Neck.score"].replace(0, np.nan)
df_cd1["Neck.score_scaled"] = df_cd1["Neck.score_scaled"].replace(0, np.nan)

# %% show histogram of Neck.score and Neck.score scaled for bl6 w/o 0
sleapf.plot_histogram_with_percentile(
    df_bl6,
    "Neck.score",
    folder_path,
    "bl6_neck_score_no_null",
    experimenter_name,
    mouse_id,
    session,
    experiment_id,
    analyst_name,
    analysis_date,
    percentile=0.05,
    num_bins=20,
)
sleapf.plot_histogram_with_percentile(
    df_bl6,
    "Neck.score_scaled",
    folder_path,
    "bl6_neck_score_scaled_no_null",
    experimenter_name,
    mouse_id,
    session,
    experiment_id,
    analyst_name,
    analysis_date,
    percentile=0.05,
    num_bins=20,
)

# %% show histogram of Neck.score and Neck.score scaled for cd1 w/o 0
sleapf.plot_histogram_with_percentile(
    df_cd1,
    "Neck.score",
    folder_path,
    "cd1_neck_score_no_null",
    experimenter_name,
    mouse_id,
    session,
    experiment_id,
    analyst_name,
    analysis_date,
    percentile=0.05,
    num_bins=20,
)
sleapf.plot_histogram_with_percentile(
    df_cd1,
    "Neck.score_scaled",
    folder_path,
    "cd1_neck_score_scaled_no_null",
    experimenter_name,
    mouse_id,
    session,
    experiment_id,
    analyst_name,
    analysis_date,
    percentile=0.05,
    num_bins=20,
)

sleapf.plot_spatial(
    df_bl6,
    df_cd1,
    "Neck.x",
    "Neck.y",
    "Neck Coordinates before threshold cleaning",
    folder_path,
    experimenter_name,
    mouse_id,
    session,
    experiment_id,
    analyst_name,
    analysis_date,
)

# %% temporal spatial plot (heatmap) of Neck.x and Neck.y coordinates for bl6 mouse + cd1 mouse
sleapf.plot_temporal_spatial(
    df_bl6,
    "Neck.x",
    "Neck.y",
    "frame_idx",
    "Bl6 Neck Coordinates before threshold cleaning",
    folder_path,
    experimenter_name,
    mouse_id,
    session,
    experiment_id,
    analyst_name,
    analysis_date,
)

# %% only necessary for mouse 2539
# TODO mouse 2539
# columns_to_check = ["Neck.x", "Neck.y"]
# df_bl6.loc[df_bl6["Neck.y"] <= 55, columns_to_check] = np.nan

# %% only necessary for mouse 2060
# TODO mouse 2060
# columns_to_check = ["Neck.x", "Neck.y"]
# df_bl6.loc[df_bl6["Neck.y"] <= 42, columns_to_check] = np.nan

# %% clean out scores below 20% percentile
# TODO change threshold if necessary
threshold_bl6 = np.nanpercentile(df_bl6["Neck.score"], 5)
threshold_cd1 = np.nanpercentile(df_cd1["Neck.score"], 5)
df_bl6, df_cd1 = sleapf.clean_data(threshold_bl6, threshold_cd1, df_bl6, df_cd1)

# %% histogram of Neck score of bl6 + cd1 mouse after threshold cleaning
sleapf.plot_histogram_without_percentile(
    df_bl6,
    "Neck.score",
    folder_path,
    "bl6_neck_score_after_cleaning",
    experimenter_name,
    mouse_id,
    session,
    experiment_id,
    analyst_name,
    analysis_date,
    num_bins=20,
)
sleapf.plot_histogram_without_percentile(
    df_cd1,
    "Neck.score",
    folder_path,
    "cd1_neck_score_after_cleaning",
    experimenter_name,
    mouse_id,
    session,
    experiment_id,
    analyst_name,
    analysis_date,
    num_bins=20,
)


# %% spatial plot of Neck.x and Neck.y coordinates for bl6 mouse + cd1 mouse
sleapf.plot_spatial(
    df_bl6,
    df_cd1,
    "Neck.x",
    "Neck.y",
    "Neck Coordinates before moving median filter",
    folder_path,
    experimenter_name,
    mouse_id,
    session,
    experiment_id,
    analyst_name,
    analysis_date,
)

# %% temporal spatial plot (heatmap) of Neck.x and Neck.y coordinates for bl6 mouse + cd1 mouse
sleapf.plot_temporal_spatial(
    df_bl6,
    "Neck.x",
    "Neck.y",
    "frame_idx",
    "Bl6 Neck Coordinates before moving median filter",
    folder_path,
    experimenter_name,
    mouse_id,
    session,
    experiment_id,
    analyst_name,
    analysis_date,
)

# %% define coordinates columns
columns_coords = ["Nose.x", "Nose.y", "Neck.x", "Neck.y", "Tail_Base.x", "Tail_Base.y"]

# %% apply moving median filter to all coordinate columns
# TODO change if necessary
df_bl6 = sleapf.moving_median_filter(df_bl6, columns_coords, window=10)  # kernel size
df_cd1 = sleapf.moving_median_filter(df_cd1, columns_coords, window=10)

# %% spatial plot of Neck.x and Neck.y coordinates for bl6 mouse + cd1 mouse
sleapf.plot_spatial(
    df_bl6,
    df_cd1,
    "Neck.x",
    "Neck.y",
    "Neck Coordinates after moving median filter",
    folder_path,
    experimenter_name,
    mouse_id,
    session,
    experiment_id,
    analyst_name,
    analysis_date,
)

# %% temporal spatial plot (heatmap) of Neck.x and Neck.y coordinates for bl6 mouse + cd1 mouse
sleapf.plot_temporal_spatial(
    df_bl6,
    "Neck.x",
    "Neck.y",
    "frame_idx",
    "Bl6 Neck Coordinates after moving median filter",
    folder_path,
    experimenter_name,
    mouse_id,
    session,
    experiment_id,
    analyst_name,
    analysis_date,
)

# %%
df_bl6[columns_coords] = df_bl6[columns_coords].interpolate(method="linear", axis=0)
df_cd1[columns_coords] = df_cd1[columns_coords].interpolate(method="linear", axis=0)
# %% interpolate missing values for all coordinate columns
# df_bl6 = sleapf.interpolate_missing_values(df_bl6, columns_coords)
# df_cd1 = sleapf.interpolate_missing_values(df_cd1, columns_coords)

# %% spatial plot of Neck.x and Neck.y coordinates for bl6 mouse + cd1 mouse
sleapf.plot_spatial(
    df_bl6,
    df_cd1,
    "Neck.x",
    "Neck.y",
    "Neck Coordinates after linear interpolation",
    folder_path,
    experimenter_name,
    mouse_id,
    session,
    experiment_id,
    analyst_name,
    analysis_date,
)

# %% temporal spatial plot (heatmap) of Neck.x and Neck.y coordinates for bl6 mouse + cd1 mouse
sleapf.plot_temporal_spatial(
    df_bl6,
    "Neck.x",
    "Neck.y",
    "frame_idx",
    "Bl6 Neck Coordinates after linear interpolation",
    folder_path,
    experimenter_name,
    mouse_id,
    session,
    experiment_id,
    analyst_name,
    analysis_date,
)  # plt.scatter only plots individual data points so it is expexted that the lines are not continuously


# %%
# plt.plot(df_cd1["frame_idx"], df_cd1["Nose.x"], color="orange")
# plt.plot(df_bl6["frame_idx"], df_bl6["Nose.x"], color="blue")
# plt.axhline(y=x_bound_left_cd1, color="red", linestyle="--")
# plt.axhline(y=x_bound_right_cd1, color="red", linestyle="--")


# %% show rectangle to make sure it was drawn correctly, close this window independently if you're happy with the drawn ROI
# sleapf.display_rectangle(video_path, frame_number, xleftT, yleftT, xrightB, yrightB)


# %% create empty tensor
mask_frames_bl6 = sleapf.empty_frame_tensor(video_path)

# %%
# all x-column stored in dimension 1 of tensor
columns_dim1 = ["Nose.x", "Neck.x", "Tail_Base.x"]

# all y-column stored in dimension 2 of tensor
columns_dim2 = ["Nose.y", "Neck.y", "Tail_Base.y"]

# Assign values from DataFrame to the tensor
mask_frames_bl6[:, :, 0] = df_bl6[columns_dim1].to_numpy()
mask_frames_bl6[:, :, 1] = df_bl6[columns_dim2].to_numpy()

# %%
x_coords_bl6 = mask_frames_bl6[:, :, 0]
y_coords_bl6 = mask_frames_bl6[:, :, 1]

# boolean vector whether body parts from bl6 mouse are inside ROI
x_mask_bl6 = np.logical_and(x_coords_bl6 >= xleftT, x_coords_bl6 <= xrightB)
y_mask_bl6 = np.logical_and(y_coords_bl6 >= yrightB, y_coords_bl6 <= yleftT)

# assign boolean vectors to tensor
mask_frames_bl6[:, :, 2] = x_mask_bl6
mask_frames_bl6[:, :, 3] = y_mask_bl6

# %%
# bl6 mouse in roi when both boolean vector for x and boolean vector for y coordinate True
in_roi = np.logical_and(x_mask_bl6 == 1, y_mask_bl6 == 1)

# gives you row names/index, ergo frame numbers for Neck --> Neck is second body part in tensor --> in_roi[:, 1]
frame_idx_interaction = np.where(in_roi[:, 1] == 1)[0]
# total number of frames Neck in roi
total_frames_interaction = np.sum(in_roi[:, 1] == 1)

# %% show 10 random frames to check whether it worked
# random_frames = np.random.choice(frame_idx_values_int64, size=9, replace=False)
random_frames = np.random.choice(frame_idx_interaction, size=9, replace=False)

# %%
np.vectorize(sleapf.rectangle_and_save)(
    video_path,
    random_frames,
    xleftT,
    yleftT,
    xrightB,
    yrightB,
    folder_path,
    experimenter_name,
    mouse_id,
    session,
    experiment_id,
    analyst_name,
    analysis_date,
)

# %%
# Read all saved frames and store them in a list
file_paths = [
    os.path.join(
        folder_path,
        f"{experimenter_name}_{mouse_id}_{session}_{experiment_id}_{analyst_name}_{analysis_date}_frame{frameNumber}.jpg",
    )
    for frameNumber in random_frames
]
# Read the frames
saved_frames = [cv2.imread(file_path) for file_path in file_paths]

# %%
sleapf.plot_frames_control(
    saved_frames,
    random_frames,
    folder_path,
    "roi",
    experimenter_name,
    mouse_id,
    session,
    experiment_id,
    analyst_name,
    analysis_date,
)


# %%
df_in_roi = pd.DataFrame(in_roi)

df_in_roi.columns = ["Nose", "Neck", "Tail_Base"]

df_in_roi = df_in_roi.rename_axis("frame_idx")  # rename index to frame_idx

# %%
csv_file_path = os.path.join(
    folder_path,
    f"{experimenter_name}_{mouse_id}_{session}_{experiment_id}_{analyst_name}_{analysis_date}_in_roi.csv",
)

# Save the DataFrame to CSV
df_in_roi.to_csv(csv_file_path)

# %% calculate s in roi
s_in_roi = total_frames_interaction / fps

# %%
# Load the DataFrame with coordinates (assuming you have loaded df_bl6 and df_cd1)
v_x_neck_bl6 = df_bl6["Neck.x"].values
v_y_neck_bl6 = df_bl6["Neck.y"].values
v_x_nose_bl6 = df_bl6["Nose.x"].values
v_y_nose_bl6 = df_bl6["Nose.y"].values
v_x_neck_cd1 = df_cd1["Neck.x"].values
v_y_neck_cd1 = df_cd1["Neck.y"].values
# Vectors from origin
x_origin = 0
y_origin = 0

# Calculate the vectors
o_neck_bl6 = np.array([v_x_neck_bl6 - x_origin, v_y_neck_bl6 - y_origin]).T
o_nose_bl6 = np.array([v_x_nose_bl6 - x_origin, v_y_nose_bl6 - y_origin]).T
o_neck_cd1 = np.array([v_x_neck_cd1 - x_origin, v_y_neck_cd1 - y_origin]).T

vector_neck_nose = o_nose_bl6 - o_neck_bl6
vector_nose_neck_cd1 = o_nose_bl6 - o_neck_cd1

# Calculate the angle between the vectors
angles = sleapf.angle_between_vectors(vector_neck_nose, vector_nose_neck_cd1)

# Calculate the cross product for every frame
cross_products = sleapf.calculate_cross_product(vector_neck_nose, vector_nose_neck_cd1)


# %%
df_in_roi_ang_dir = df_in_roi.copy()
# temp['angles'] = angles
# temp['direction'] = cross_products

cond_degree = 160  # 180 - +- 20 degrees --> mice have 40° field of vision
cond_angles = angles >= cond_degree
df_in_roi_ang_dir["cond_angles"] = cond_angles

cond_direction = 0
cond_direction = cross_products > cond_direction
# cond_direction = (cross_products > 0) & (cross_products < 10000)
df_in_roi_ang_dir["cond_direction"] = cond_direction

# %%
frame_idx_roi_ang_dir = np.where(
    (df_in_roi_ang_dir["cond_angles"] == 1)
    & (df_in_roi_ang_dir["cond_direction"] == 1)
    & (df_in_roi_ang_dir["Nose"] == 1)
    & (df_in_roi_ang_dir["Neck"] == 1)
)[
    0
]  # which frames

total_frames_roi_ang_dir = np.where(
    (df_in_roi_ang_dir["cond_angles"] == 1)
    & (df_in_roi_ang_dir["cond_direction"] == 1)
    & (df_in_roi_ang_dir["Nose"] == 1)
    & (df_in_roi_ang_dir["Neck"] == 1)
)[0].shape[
    0
]  # number of frames for which this is true


# %%
# frameN = frame_idx_roi_ang_dir[0]
# sleapf.draw_frame_with_vector(
#     video_path, df_bl6, df_cd1, frameN, xleftT, yleftT, xrightB, yrightB
# )


# %%
random_frames_roi_ang_dir = np.random.choice(
    frame_idx_roi_ang_dir, size=20, replace=False
)

# %%
x_neck_bl6 = df_bl6["Neck.x"][random_frames_roi_ang_dir]
y_neck_bl6 = df_bl6["Neck.y"][random_frames_roi_ang_dir]
x_nose_bl6 = df_bl6["Nose.x"][random_frames_roi_ang_dir]
y_nose_bl6 = df_bl6["Nose.y"][random_frames_roi_ang_dir]

x_neck_cd1 = df_cd1["Neck.x"][random_frames_roi_ang_dir]
y_neck_cd1 = df_cd1["Neck.y"][random_frames_roi_ang_dir]
# %%
np.vectorize(sleapf.rectangle_ang_dir_and_save)(
    video_path,
    random_frames_roi_ang_dir,
    xleftT,
    yleftT,
    xrightB,
    yrightB,
    folder_path,
    experimenter_name,
    mouse_id,
    session,
    experiment_id,
    analyst_name,
    analysis_date,
    x_neck_bl6,
    y_neck_bl6,
    x_nose_bl6,
    y_nose_bl6,
    x_neck_cd1,
    y_neck_cd1,
    origin=False,
    crop_margin=100,
)

# %%
# Read all saved frames and store them in a list
file_paths_roi_ang_dir = [
    os.path.join(
        folder_path,
        f"{experimenter_name}_{mouse_id}_{session}_{experiment_id}_{analyst_name}_{analysis_date}_roi_ang_dir_frame{frameNumber}.jpg",
    )
    for frameNumber in random_frames_roi_ang_dir
]
# Read the frames
saved_frames_roi_ang_dir = [
    cv2.imread(file_path) for file_path in file_paths_roi_ang_dir
]

# %%
sleapf.plot_frames_control(
    saved_frames_roi_ang_dir,
    random_frames_roi_ang_dir,
    folder_path,
    "roi_ang_dir",
    experimenter_name,
    mouse_id,
    session,
    experiment_id,
    analyst_name,
    analysis_date,
    5,
    4,
)

# %%
csv_file_path_roi_ang_dir = os.path.join(
    folder_path,
    f"{experimenter_name}_{mouse_id}_{session}_{experiment_id}_{analyst_name}_{analysis_date}_in_roi_ang_dir.csv",
)

# Save the DataFrame to CSV
df_in_roi_ang_dir.to_csv(csv_file_path_roi_ang_dir)


# %%
s_in_roi_ang_dir = total_frames_roi_ang_dir / fps

# %%
# List that we want to add as a new row
header = ["mouse_id", "session", "s_in_roi", "s_in_roi_ang_dir"]
seconds_info = [mouse_id, session, s_in_roi, s_in_roi_ang_dir]

# Check if the CSV file exists
file_exists = os.path.isfile("5pc_nz_td_id_seconds_info.csv")

# Open our existing CSV file in append mode
# Create a file object for this file
# nz for new zone analysis
with open("5pc_nz_td_id_seconds_info.csv", "a") as f_object:  # a is for append

    # Pass this file object to csv.writer() and get a writer object
    writer_object = writer(f_object)

    # Write the header row if the file doesn't exist
    if not file_exists:
        writer_object.writerow(header)

    # Write the data row
    writer_object.writerow(seconds_info)
