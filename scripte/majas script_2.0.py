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
script_dir = os.path.dirname(os.path.abspath(__file__))
# Set the current working directory to the script directory
os.chdir(script_dir)

# %% import analysis functions
import td_id_analysis_sleap_svenja_functions as sleapf


# %% for quicker things
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
mouse_id = experiment_info["Mouse ID"]
session = 1
experiment_id = "SoIn"
analyst_name = "MN"
# TODO change cohort
cohort = "coh4_str"
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
csv_path = os.path.join(
    folder_path, f"model_td_id_{cohort}_{mouse_id}_session{session}.analysis.csv"
)

# %%
vid = cv2.VideoCapture(video_path)
total_frame_count = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
fps = vid.get(cv2.CAP_PROP_FPS)

# %%
# TODO: change according to progress.xlsx
frame_start = 1

# %%
df_bl6 = pd.read_csv(csv_path)

# %%
df_bl6 = sleapf.add_missing_frames(df_bl6, total_frame_count - 1)

# fill track column with 'bl6'
df_bl6["track"].fillna("bl6", inplace=True)

# %%
# cut-off all the frames where mouse wasn't even in the video yet
df_bl6 = df_bl6[df_bl6["frame_idx"] >= (frame_start - 1)]

# reset index
df_bl6.reset_index(drop=True, inplace=True)

# %% create new normalised column (range 0-1), scaled by min and max possible values
df_bl6 = sleapf.scale_column(df_bl6, "Neck.score", 0, 1)
# %% name attribute for plot
df_bl6.name = "bl6"

# %%
# histogram of Neck score of bl6 mouse
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
# histogram of Neck score scaled of bl6 mouse
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
# %% replace zeros in 'Neck.score' column with nan
df_bl6["Neck.score"] = df_bl6["Neck.score"].replace(0, np.nan)
df_bl6["Neck.score_scaled"] = df_bl6["Neck.score_scaled"].replace(0, np.nan)

# %%
# histogram of Neck score of bl6 mouse w/o 0
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
# histogram of Neck score scaled of bl6 mouse w/o 0
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

# %% spatial plot of Neck.x and Neck.y coordinates for bl6 mouse
sleapf.plot_spatial_sess1(
    df_bl6,
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

# %% temporal spatial plot (heatmap) of Neck.x and Neck.y coordinates for bl6 mouse
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

# %% clean out scores below 20% percentile
# TODO if necessary, adjust threshold
threshold_bl6 = np.nanpercentile(df_bl6["Neck.score"], 5)
# returns df_bl6 columns without track and frame_idx
columns_to_clean = df_bl6.columns.difference(["track", "frame_idx"])
df_bl6.loc[df_bl6["Neck.score"] <= threshold_bl6, columns_to_clean] = np.nan
# %%
# histogram of Neck score of bl6 mouse after threshold cleaning
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

# %% spatial plot of Neck.x and Neck.y coordinates for bl6 mouse
sleapf.plot_spatial_sess1(
    df_bl6,
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

# %% temporal spatial plot (heatmap) of Neck.x and Neck.y coordinates for bl6 mouse
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
df_bl6 = sleapf.moving_median_filter(df_bl6, columns_coords, window=10)  # kernel size


# %% spatial plot of Neck.x and Neck.y coordinates for bl6 mouse
sleapf.plot_spatial_sess1(
    df_bl6,
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

# %% temporal spatial plot (heatmap) of Neck.x and Neck.y coordinates for bl6 mouse
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
# %% interpolate missing values for all coordinate columns
df_bl6[columns_coords] = df_bl6[columns_coords].interpolate(method="linear", axis=0)
# df_bl6 = sleapf.interpolate_missing_values(df_bl6, columns_coords)

# %% spatial plot of Neck.x and Neck.y coordinates for bl6 mouse
sleapf.plot_spatial_sess1(
    df_bl6,
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
# %% temporal spatial plot (heatmap) of Neck.x and Neck.y coordinates for bl6 mouse
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
)

# %%
df_video_info = pd.read_csv(r"D:\nz_td_id_video_info.csv")
df_video_info_coh = df_video_info[df_video_info["cohort"] == cohort]

# %% get boundaries of ROI
xleftT = df_video_info_coh.xleftT.values[0]
yleftT = df_video_info_coh.yleftT.values[0]
xrightB = df_video_info_coh.xrightB.values[0]
yrightB = df_video_info_coh.yrightB.values[0]

# %% show rectangle to make sure it was drawn correctly, close this window independently if you're happy with the drawn ROI
# sleapf.display_rectangle(video_path, frame_number, xleftT, yleftT, xrightB, yrightB)

# %% create empty tensor
mask_frames_bl6 = np.zeros(
    (int(len(df_bl6)), 3, 4), np.float64
)  # 3 cause 3 skeleton nodes, 4 cause 2 dimensions (x/y) x 2 boolean (True/False)

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

# gives you row names/index, ergo frame numbers for neck --> neck is second body part in tensor --> in_roi[:, 0]
frame_idx_interaction = np.where(in_roi[:, 1] == 1)[0]
# total number of frames neck in roi, same as len(np.where(in_roi[:, 1] == 1)[0])
total_frames_interaction = np.where(in_roi[:, 1] == 1)[0].shape[0]

# %% create 9 random frames where bl6 mouse is in ROI
random_frame_idx = np.random.choice(frame_idx_interaction, size=9, replace=False)
random_frames = df_bl6.loc[random_frame_idx, "frame_idx"].values
# %% save each of the 9 frames as a .jpg file with ROI rectangle
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
# get file paths of all 9 saved frames and store them in a list
file_paths = [
    os.path.join(
        folder_path,
        f"{experimenter_name}_{mouse_id}_{session}_{experiment_id}_{analyst_name}_{analysis_date}_frame{frameNumber}.jpg",
    )
    for frameNumber in random_frames
]
# Read the frames
saved_frames = [cv2.imread(file_path) for file_path in file_paths]

# %% plot 9 random frames as a control/sanity check
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
# create dataframe
df_in_roi = pd.DataFrame(in_roi)

# rename value columns
df_in_roi.columns = ["Nose", "Neck", "Tail_Base"]

# rename index column
df_in_roi = df_in_roi.rename_axis("frame_idx")

# create additional column
df_in_roi["frame_number"] = df_bl6["frame_idx"].values
# %%
csv_file_path = os.path.join(
    folder_path,
    f"{experimenter_name}_{mouse_id}_{session}_{experiment_id}_{analyst_name}_{analysis_date}_in_roi.csv",
)

# Save in_roi dataframe as .csv file
df_in_roi.to_csv(csv_file_path)


# %% calculate s in roi
s_in_roi = total_frames_interaction / fps

# %% get coordinates for calculating the middle of the box
x_bound_left_cd1 = df_video_info_coh.x_bound_left_cd1.values[0]
x_bound_right_cd1 = df_video_info_coh.x_bound_right_cd1.values[0]
y_bound_bl6 = df_video_info_coh.y_bound_bl6.values[0]

midB = np.zeros((2), dtype=np.float64)
midB[0] = (x_bound_left_cd1 + x_bound_right_cd1) / 2  # x coordinates
midB[1] = y_bound_bl6

midB[1] = midB[1] * (2 / 3)


# %%
# get x- and y- values of neck and nose of bl6 mouse
v_x_neck_bl6 = df_bl6["Neck.x"].values
v_y_neck_bl6 = df_bl6["Neck.y"].values
v_x_nose_bl6 = df_bl6["Nose.x"].values
v_y_nose_bl6 = df_bl6["Nose.y"].values

# get x- and y-value of middle of the box
v_x_mid_box = midB[0]
v_y_mid_box = midB[1]

# x- and y- coordinates of origin
x_origin = 0
y_origin = 0

# calculate origin vectors
o_neck_bl6 = np.array([v_x_neck_bl6 - x_origin, v_y_neck_bl6 - y_origin]).T
o_nose_bl6 = np.array([v_x_nose_bl6 - x_origin, v_y_nose_bl6 - y_origin]).T
o_mid_box = np.array([v_x_mid_box - x_origin, v_y_mid_box - y_origin]).T

# calculate neck-nose and nose-mid_box vectors
vector_neck_nose = o_nose_bl6 - o_neck_bl6
vector_nose_mid_box = o_nose_bl6 - o_mid_box

# calculate the angle between the vectors
angles = sleapf.angle_between_vectors(vector_neck_nose, vector_nose_mid_box)

# calculate the cross product for every frame
cross_products = sleapf.calculate_cross_product(vector_neck_nose, vector_nose_mid_box)


# %%
# create df for in ROI + angle and direction
df_in_roi_ang_dir = df_in_roi.copy()

# angle condition
cond_degree = 160  # 180 - 20 degrees --> mice have 40° field of vision --> 40°/2 = 20°
cond_angles = angles >= cond_degree
df_in_roi_ang_dir["cond_angles"] = cond_angles

# direction condition
cond_direction = 0
cond_direction = cross_products > cond_direction
df_in_roi_ang_dir["cond_direction"] = cond_direction

# %%
frame_idx_roi_ang_dir = np.where(
    (df_in_roi_ang_dir["cond_angles"] == 1)
    & (df_in_roi_ang_dir["cond_direction"] == 1)
    & (df_in_roi_ang_dir["Nose"] == 1)
    & (df_in_roi_ang_dir["Neck"] == 1)
)[
    0
]  # which frames meet requirements/conditions

total_frames_roi_ang_dir = np.where(
    (df_in_roi_ang_dir["cond_angles"] == 1)
    & (df_in_roi_ang_dir["cond_direction"] == 1)
    & (df_in_roi_ang_dir["Nose"] == 1)
    & (df_in_roi_ang_dir["Neck"] == 1)
)[0].shape[
    0
]  # number of frames for which conditions are true

# %% create 20 random frames where bl6 mouse is in ROI + ang and direction conditions met
random_frame_idx = np.random.choice(frame_idx_roi_ang_dir, size=20, replace=False)
random_frames_roi_ang_dir = df_in_roi_ang_dir["frame_number"][random_frame_idx].values

# %% x- and y-coordinates of neck and nose for the 20 random frames
x_neck_bl6 = df_bl6["Neck.x"][random_frame_idx]
y_neck_bl6 = df_bl6["Neck.y"][random_frame_idx]
x_nose_bl6 = df_bl6["Nose.x"][random_frame_idx]
y_nose_bl6 = df_bl6["Nose.y"][random_frame_idx]


# %% save each of the 20 frames as a .jpg file with ROI rectangle + vectors neck nose and nose mid_box
np.vectorize(sleapf.rectangle_ang_dir_and_save_sess1)(
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
    v_x_mid_box,
    v_y_mid_box,
    origin=False,
    crop_margin=100,
)

# %%
# get file paths of all 20 saved frames and store them in a list
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

# %% plot 20 random frames as a control/sanity check
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

# Save in_roi_ang_dir dataframe as .csv file
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
# %%
