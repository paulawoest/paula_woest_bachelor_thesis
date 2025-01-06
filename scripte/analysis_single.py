# %% importing libraries
import cv2
import numpy as np
import os
import pandas as pd
from csv import writer
from datetime import date

# import analysis functions
import td_id_analysis_sleap_svenja_functions as sleapf

# %%
# directory where this script/file is saved
script_dir = os.path.dirname(os.path.abspath(__file__))
# Set the current working directory to the script directory
os.chdir(script_dir)

# %%
experimenter_name = "SB"
mouse_id = int(input("Enter mouse ID: "))
cohort_number = int(input("Enter cohort number: "))  # i.e. either 6 or 7
session = 1
experiment_id = "uSoIn"
analyst_name = "PW"
cohort = f"coh_{cohort_number}"
analysis_date = date.today()

# %% set folder path where files etc. get saved
folder_path = os.path.join(os.getcwd(), str(mouse_id), f"session{session}")

#  select file paths
csv_path = os.path.join(
    folder_path, f"model_td_grayscale_{cohort}_{mouse_id}_session{session}.csv"
)
video_path = os.path.join(
    folder_path, f"grayscale_{cohort}_{mouse_id}_session{session}.mp4"
)
vid = cv2.VideoCapture(video_path)
total_frame_count = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
fps = vid.get(cv2.CAP_PROP_FPS)

# %%
frame_start = 1

# %%
# List that we want to add as a new row
header = ["mouse_id", "session", "fps", "frame_start"]
fps_info = [mouse_id, session, fps, frame_start]

# Check if the CSV file exists
file_exists = os.path.isfile("fps_info_sess1.csv")

# Open our existing CSV file in append mode
# Create a file object for this file
# nz for new zone analysis
with open("fps_info_sess1.csv", "a") as f_object:  # a is for append

    # Pass this file object to csv.writer() and get a writer object
    writer_object = writer(f_object)

    # Write the header row if the file doesn't exist
    if not file_exists:
        writer_object.writerow(header)

    # Write the data row
    writer_object.writerow(fps_info)

# %%
df_video_info = pd.read_csv(os.path.join(os.getcwd(), "video_info.csv"))
df_video_info_coh = df_video_info[df_video_info["cohort"] == cohort_number]

# %% get boundaries of ROI
xleftT = df_video_info_coh.xleftT.values[0]
yleftT = df_video_info_coh.yleftT.values[0]
xrightB = df_video_info_coh.xrightB.values[0]
yrightB = df_video_info_coh.yrightB.values[0]

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

# %%
# create new normalised column (range 0-1), scaled by min and max possible values
df_bl6 = sleapf.scale_column(df_bl6, "Neck.score", 0, 1)


# name attribute for plot
df_bl6.name = "bl6"


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

# %% replace zeros in 'Neck.score' column with nan
df_bl6["Neck.score"] = df_bl6["Neck.score"].replace(0, np.nan)
df_bl6["Neck.score_scaled"] = df_bl6["Neck.score_scaled"].replace(0, np.nan)

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

# %% clean out scores below 20% percentile
# TODO change threshold if necessary
threshold_bl6 = np.nanpercentile(df_bl6["Neck.score"], 5)
# returns df_bl6 columns without track and frame_idx
columns_to_clean = df_bl6.columns.difference(["track", "frame_idx"])
df_bl6.loc[df_bl6["Neck.score"] <= threshold_bl6, columns_to_clean] = np.nan

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


# %% create empty tensor
mask_frames_bl6 = np.zeros(
    (int(len(df_bl6)), 3, 4), np.float64
)  # 3 cause 3 skeleton nodes, 4 cause 2 dimensions (x/y) x 2 boolean (True/False)


# %%
# all x-column stored in dimension 1 of tensor
columns_x_dim = ["Nose.x", "Neck.x", "Tail_Base.x"]

# all y-column stored in dimension 2 of tensor
columns_y_dim = ["Nose.y", "Neck.y", "Tail_Base.y"]

# Assign values from DataFrame to the tensor
mask_frames_bl6[:, :, 0] = df_bl6[columns_x_dim].to_numpy()
mask_frames_bl6[:, :, 1] = df_bl6[columns_y_dim].to_numpy()

# %%
x_coords_bl6 = mask_frames_bl6[:, :, 0]
y_coords_bl6 = mask_frames_bl6[:, :, 1]

# boolean vector whether body parts from bl6 mouse are inside ROI
x_mask_bl6 = np.logical_and(x_coords_bl6 >= xleftT, x_coords_bl6 <= xrightB)
y_mask_bl6 = np.logical_and(y_coords_bl6 <= yrightB, y_coords_bl6 >= yleftT)

# assign boolean vectors to tensor
mask_frames_bl6[:, :, 2] = x_mask_bl6
mask_frames_bl6[:, :, 3] = y_mask_bl6

# %% #TODO in roi
# bl6 mouse in roi when both boolean vector for x and boolean vector for y coordinate True
in_roi = np.logical_and(x_mask_bl6 == 1, y_mask_bl6 == 1)

# gives you row names/index, ergo frame numbers for Neck --> Neck is second body part in tensor --> in_roi[:, 1]
frame_idx_interaction = np.where(in_roi[:, 1] == 1)[0]
# total number of frames Neck in roi
total_frames_interaction = np.sum(in_roi[:, 1] == 1)

df_in_roi = pd.DataFrame(in_roi)

df_in_roi.columns = ["Nose", "Neck", "Tail_Base"]

df_in_roi = df_in_roi.rename_axis("frame_idx")  # rename index to frame_idx

csv_file_path = os.path.join(
    folder_path,
    f"{experimenter_name}_{mouse_id}_{session}_{experiment_id}_{analyst_name}_{analysis_date}_in_roi.csv",
)

# Save the DataFrame to CSV
df_in_roi.to_csv(csv_file_path)

# %% calculate s in roi
s_in_roi = total_frames_interaction / fps

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

# %% get coordinates for calculating the middle of the box
x_bound_left_cd1 = df_video_info_coh.x_bound_left_cd1.values[0]
x_bound_right_cd1 = df_video_info_coh.x_bound_right_cd1.values[0]
y_bound_bl6 = df_video_info_coh.y_bound_bl6.values[0]

midB = np.zeros((2), dtype=np.float64)
midB[0] = (x_bound_left_cd1 + x_bound_right_cd1) / 2  # x coordinates
midB[1] = y_bound_bl6

midB[1] = midB[1] * (2 / 3)

# %% #TODO in roi ang dir
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
vector_neck_bl6_nose_bl6 = o_nose_bl6 - o_neck_bl6
vector_neck_bl6_mid_box = o_neck_bl6 - o_mid_box

# calculate the angle between the vectors
angles = sleapf.angle_between_vectors(vector_neck_bl6_nose_bl6, vector_neck_bl6_mid_box)

# calculate the cross product for every frame
cross_products = np.cross(vector_neck_bl6_nose_bl6, vector_neck_bl6_mid_box)

# %%
df_in_roi_ang_dir = df_in_roi.copy()
# angle condition
cond_degree = 20  # mice have 40° field of vision --> 40°/2 = 20°
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
)[
    0
]  # which frames meet requirements/conditions

total_frames_roi_ang_dir = np.where(
    (df_in_roi_ang_dir["cond_angles"] == 1)
    & (df_in_roi_ang_dir["cond_direction"] == 1)
    & (df_in_roi_ang_dir["Nose"] == 1)
)[0].shape[
    0
]  # number of frames for which conditions are true

# Save in_roi_ang_dir dataframe as .csv file
csv_file_path_roi_ang_dir = os.path.join(
    folder_path,
    f"{experimenter_name}_{mouse_id}_{session}_{experiment_id}_{analyst_name}_{analysis_date}_in_roi_ang_dir.csv",
)

# Save the DataFrame to CSV
df_in_roi_ang_dir.to_csv(csv_file_path_roi_ang_dir)

# %%
s_in_roi_ang_dir = total_frames_roi_ang_dir / fps

# %%
random_frames_roi_ang_dir = np.random.choice(
    frame_idx_roi_ang_dir, size=20, replace=False
)

# %% show 10 random frames to check whether it worked
x_neck_bl6 = df_bl6["Neck.x"][random_frames_roi_ang_dir]
y_neck_bl6 = df_bl6["Neck.y"][random_frames_roi_ang_dir]
x_nose_bl6 = df_bl6["Nose.x"][random_frames_roi_ang_dir]
y_nose_bl6 = df_bl6["Nose.y"][random_frames_roi_ang_dir]

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
# List that we want to add as a new row
header = ["mouse_id", "session", "s_in_roi", "s_in_roi_ang_dir"]
seconds_info = [mouse_id, session, s_in_roi, s_in_roi_ang_dir]

# Check if the CSV file exists
file_exists = os.path.isfile("seconds_info.csv")

# Open our existing CSV file in append mode
# Create a file object for this file
# nz for new zone analysis
with open("seconds_info.csv", "a") as f_object:  # a is for append

    # Pass this file object to csv.writer() and get a writer object
    writer_object = writer(f_object)

    # Write the header row if the file doesn't exist
    if not file_exists:
        writer_object.writerow(header)

    # Write the data row
    writer_object.writerow(seconds_info)
