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
session = 2
experiment_id = "uSoIn"
analyst_name = "PW"
cohort = f"coh_{cohort_number}"
analysis_date = date.today()

# %% set folder path where files etc. get saved
folder_path = os.path.join(os.getcwd(), str(mouse_id), f"session{session}")

#  select file paths
csv_path_uncleaned = os.path.join(
    folder_path, f"model_td_grayscale_{cohort}_{mouse_id}_session{session}.csv"
)
video_path = os.path.join(
    folder_path, f"grayscale_{cohort}_{mouse_id}_session{session}.mp4"
)
vid = cv2.VideoCapture(video_path)
total_frame_count = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
fps = vid.get(cv2.CAP_PROP_FPS)

# %%
# List that we want to add as a new row
header = ["mouse_id", "session", "fps"]
fps_info = [mouse_id, session, fps]

# Check if the CSV file exists
file_exists = os.path.isfile("fps_info.csv")

# Open our existing CSV file in append mode
# Create a file object for this file
# nz for new zone analysis
with open("fps_info.csv", "a") as f_object:  # a is for append

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

# %% fill dataframe with missing frames
df, df_filled = sleapf.fill_df(csv_path_uncleaned)


# %% #TODO masks for when tracks are switched
# indices where bl6 is outside of bounds, i.e., inside the box
mask_bl6 = sleapf.create_mask_bl6(
    df_filled, x_bound_left_bl6, x_bound_right_bl6, y_bound_bl6
)
# indices where cd1 is outside of bounds, i.e., outside the box
mask_cd1 = sleapf.create_mask_cd1(
    df_filled, x_bound_left_cd1, x_bound_right_cd1, y_bound_cd1
)

# frame_idx where bl6 mouse outside of bounds
bl6_outside_f_indices = df_filled[(mask_bl6)].frame_idx
# df with rows where bl6 mouse is outside of bounds + corresponding cd1 rows (same frame idx)
df_bl6_outside_cd1 = df_filled[df_filled["frame_idx"].isin(bl6_outside_f_indices)]
# only cd1 tracks --> if we don't do that mask will be false due to row being bl6 track and not because conditions are met
df_bl6_outside_cd1_only = df_bl6_outside_cd1[df_bl6_outside_cd1["track"] == "cd1"]
# indices where cd1 is outside of bounds (out of the frame idx where bl6 is out of bounds, is cd1 also out of bounds)
mask_cd1_out = sleapf.create_mask_cd1(
    df_bl6_outside_cd1_only, x_bound_left_cd1, x_bound_right_cd1, y_bound_cd1
)

# frame_idx where cd1 mouse outside of bounds
cd1_outside_f_indices = df_filled[(mask_cd1)].frame_idx
# df with rows where bcd1l6 mouse is outside of bounds + corresponding bl6 rows (same frame idx)
df_cd1_outside_bl6 = df_filled[df_filled["frame_idx"].isin(cd1_outside_f_indices)]
# only bl6 tracks --> if we don't do that mask will be false due to row being cd1 track and not because conditions are met
df_cd1_outside_bl6_only = df_cd1_outside_bl6[df_cd1_outside_bl6["track"] == "bl6"]
# indices where bl6 is outside of bounds (out of the frame idx where cd1 is out of bounds, is bl6 also out of bounds)
mask_bl6_out = sleapf.create_mask_bl6(
    df_cd1_outside_bl6_only, x_bound_left_bl6, x_bound_right_bl6, y_bound_bl6
)

# %% #TODO masks for when bl6 is wrong but cd1 is right
# df where bl6 is out of bounds and cd1 is inside bounds, i.e., both are inside of the box
masked_df_bl6 = df_bl6_outside_cd1_only[~mask_cd1_out]
# Extract frame_idx where instance.score is NaN --> these are the ones we want to switch (i.e., cd1 is nan but bl6 has values inside the box --> switch)
frame_idx_s_bl6 = masked_df_bl6[masked_df_bl6["instance.score"].isna()][
    "frame_idx"
].tolist()
# Extract frame_idx where instance.score is not NaN --> this is where we want to delete bl6 tracks (i.e, cd1 has values but so does bl6 inside the box --> delete)
frame_idx_d_bl6 = masked_df_bl6[~masked_df_bl6["instance.score"].isna()][
    "frame_idx"
].tolist()

# %% #TODO masks for when cd1 is wrong but bl6 is right
# df where cd1 is out of bounds and bl6 is inside bounds, i.e., both are outside of the box
masked_df_cd1 = df_cd1_outside_bl6_only[~mask_bl6_out]
# Extract frame_idx where instance.score is NaN --> these are the ones we want to switch (i.e., bl6 is nan but cd1 has values outside the box --> switch)
frame_idx_s_cd1 = masked_df_cd1[masked_df_cd1["instance.score"].isna()][
    "frame_idx"
].tolist()
# Extract frame_idx where instance.score is not NaN --> this is where we want to delete cd1 tracks (i.e, bl6 has values but so does cd1 outside the box --> delete)
frame_idx_d_cd1 = masked_df_cd1[~masked_df_cd1["instance.score"].isna()][
    "frame_idx"
].tolist()

# create new df so we don't overwrite the old
df_new = df_filled.copy()

# %% #TODO delete + switching actions
# delete wrong bl6 tracks
# df with frames to delete
df_deleted_frames = df_filled[df_filled["frame_idx"].isin(frame_idx_d_bl6)].copy()
for frame_number in df_deleted_frames["frame_idx"].unique():
    # Get the indices of rows corresponding to the current frame number
    indices = df_deleted_frames[df_deleted_frames["frame_idx"] == frame_number].index
    # set row to nan for bl6 track
    for index in indices:
        current_track = df_new.at[index, "track"]
        if current_track == "bl6":
            df_new.iloc[index, 2:] = np.nan

# delete wrong cd1 tracks
# df with frames to delete
df_deleted_frames_cd1 = df_filled[df_filled["frame_idx"].isin(frame_idx_d_cd1)].copy()
for frame_number in df_deleted_frames_cd1["frame_idx"].unique():
    # Get the indices of rows corresponding to the current frame number
    indices = df_deleted_frames_cd1[
        df_deleted_frames_cd1["frame_idx"] == frame_number
    ].index
    # Switch track labels for both rows
    for index in indices:
        current_track = df_new.at[index, "track"]
        if current_track == "cd1":
            df_new.iloc[index, 2:] = np.nan

# switch wrong tracks
frame_idx_s = frame_idx_s_bl6 + frame_idx_s_cd1
# df with frames to switch
df_switched_frames = df_filled[df_filled["frame_idx"].isin(frame_idx_s)].copy()
for frame_number in df_switched_frames["frame_idx"].unique():
    # Get the indices of rows corresponding to the current frame number
    indices = df_switched_frames[df_switched_frames["frame_idx"] == frame_number].index
    # Switch track labels for both rows
    for index in indices:
        current_track = df_new.at[index, "track"]
        if current_track == "bl6":
            df_new.at[index, "track"] = "cd1"
        elif current_track == "cd1":
            df_new.at[index, "track"] = "bl6"

# %% now we still have cases where bl6 and cd1 both have values (code before was for when only one track had scores)
# indices where cd1 is outside of bounds
mask_cd1_v = sleapf.create_mask_cd1(
    df_new, x_bound_left_cd1, x_bound_right_cd1, y_bound_cd1
)

# indices where bl6 is outside of bounds
mask_bl6_v = sleapf.create_mask_bl6(
    df_new, x_bound_left_bl6, x_bound_right_bl6, y_bound_bl6
)

# indices where either cd1 or bl6 is out of bounds
merged_mask = mask_cd1_v | mask_bl6_v

# switch remaining frames
df_filled_switch = sleapf.switch_tracks_initial(df_new, merged_mask)

# %% spatial plot of bl6 neck before and after track switching + boundary lines
sleapf.plot_spatial_switch(
    df_filled[df_filled["track"] == "bl6"],
    "Neck.x",
    "Neck.y",
    "neck bl6 before initial cleaning",
    x_bound_left_cd1,
    x_bound_right_cd1,
    y_bound_cd1,
    x_bound_left_bl6,
    x_bound_right_bl6,
    y_bound_bl6,
    folder_path,
    experimenter_name,
    mouse_id,
    session,
    experiment_id,
    analyst_name,
    analysis_date,
)
sleapf.plot_spatial_switch(
    df_filled_switch[df_filled_switch["track"] == "bl6"],
    "Neck.x",
    "Neck.y",
    "neck bl6 after initial cleaning",
    x_bound_left_cd1,
    x_bound_right_cd1,
    y_bound_cd1,
    x_bound_left_bl6,
    x_bound_right_bl6,
    y_bound_bl6,
    folder_path,
    experimenter_name,
    mouse_id,
    session,
    experiment_id,
    analyst_name,
    analysis_date,
)

# %% spatial plot of cd1 neck before and after track switching + boundary lines
sleapf.plot_spatial_switch(
    df_filled[df_filled["track"] == "cd1"],
    "Neck.x",
    "Neck.y",
    "neck cd1 before initial cleaning",
    x_bound_left_cd1,
    x_bound_right_cd1,
    y_bound_cd1,
    x_bound_left_bl6,
    x_bound_right_bl6,
    y_bound_bl6,
    folder_path,
    experimenter_name,
    mouse_id,
    session,
    experiment_id,
    analyst_name,
    analysis_date,
)
sleapf.plot_spatial_switch(
    df_filled_switch[df_filled_switch["track"] == "cd1"],
    "Neck.x",
    "Neck.y",
    "neck cd1 after initial cleaning",
    x_bound_left_cd1,
    x_bound_right_cd1,
    y_bound_cd1,
    x_bound_left_bl6,
    x_bound_right_bl6,
    y_bound_bl6,
    folder_path,
    experimenter_name,
    mouse_id,
    session,
    experiment_id,
    analyst_name,
    analysis_date,
)

# %% small gui to correct eventually missed frames -- 90°
(
    frames_ambig_combined_v2,
    ambig_matrix_v2,
    surrounding_foi_frames_array_v2,
    action_array_v2,
    df_filled_switch_v2,
) = sleapf.clean_with_gui(
    df_filled_switch,
    y_bound_bl6,
    y_bound_cd1,
    x_bound_left_cd1,
    x_bound_right_cd1,
    90,
    video_path,
    xleftT,
    yleftT,
    xrightB,
    yrightB,
)


# %% small gui to correct eventually missed frames -- 55°
(
    frames_ambig_combined_v3,
    ambig_matrix_v3,
    surrounding_foi_frames_array_v3,
    action_array_v3,
    df_filled_switch_v3,
) = sleapf.clean_with_gui(
    df_filled_switch_v2,
    y_bound_bl6,
    y_bound_cd1,
    x_bound_left_cd1,
    x_bound_right_cd1,
    55,
    video_path,
    xleftT,
    yleftT,
    xrightB,
    yrightB,
    gui=True,
    surrounding_foi_frames_array_old=surrounding_foi_frames_array_v2,
)

# %% small gui to correct eventually missed frames -- 20°
(
    frames_ambig_combined_v4,
    ambig_matrix_v4,
    surrounding_foi_frames_array_v4,
    action_array_v4,
    df_filled_switch_v4,
) = sleapf.clean_with_gui(
    df_filled_switch_v3,
    y_bound_bl6,
    y_bound_cd1,
    x_bound_left_cd1,
    x_bound_right_cd1,
    20,
    video_path,
    xleftT,
    yleftT,
    xrightB,
    yrightB,
    gui=True,
    last=True,
    surrounding_foi_frames_array_og=surrounding_foi_frames_array_v2,
    surrounding_foi_frames_array_old=surrounding_foi_frames_array_v3,
)

# %%
# TODO: if no v2 cleaning is necessary, i.e. surrounding_foi_frames_v2 has no frames
# action_array_v2 = []
# TODO: if no v3 cleaning is necessary, i.e. surrounding_foi_frames_v3 has no new frames
# action_array_v3 = []
# TODO: if no v4 cleaning is necessary, i.e. surrounding_foi_frames_v4 has no new frames
# action_array_v4 = []


# %% depending on how much track cleaning in small gui was necessary
# TODO: if no v2 cleaning is necessary, i.e. surrounding_foi_frames_v2 has no frames
# df_filled_switch_v4 = df_filled_switch.copy()
# TODO: if no v3 cleaning is necessary, i.e. surrounding_foi_frames_v3 has no new frames
# df_filled_switch_v4 = df_filled_switch_v2.copy()
# TODO: if no v4 cleaning is necessary, i.e. surrounding_foi_frames_v4 has no new frames
# df_filled_switch_v4 = df_filled_switch_v3.copy()
# %% save action arrays for reproducibility
# Zip the arrays together to iterate over them simultaneously
zipped_arrays = zip(action_array_v2, action_array_v3, action_array_v4)

# List that we want to add as a new row
header = ["action_array_v2", "action_array_v3", "action_array_v4"]

file_name = f"{experimenter_name}_{mouse_id}_{session}_{experiment_id}_{analyst_name}_{analysis_date}_action_info.csv"
save_path = os.path.join(folder_path, file_name)

# Open the CSV file in append mode
with open(save_path, "a") as f_object:  # 'a' is for append mode

    # Create a CSV writer object
    writer_object = writer(f_object)

    # Write the header row
    writer_object.writerow(header)

    # Write each zipped element as a separate row
    for row in zipped_arrays:
        writer_object.writerow(row)

# %% spatial plot of bl6 neck after gui track switching + boundary lines
sleapf.plot_spatial_switch(
    df_filled_switch_v4[df_filled_switch_v4["track"] == "bl6"],
    "Neck.x",
    "Neck.y",
    "neck bl6 after gui cleaning",
    x_bound_left_cd1,
    x_bound_right_cd1,
    y_bound_cd1,
    x_bound_left_bl6,
    x_bound_right_bl6,
    y_bound_bl6,
    folder_path,
    experimenter_name,
    mouse_id,
    session,
    experiment_id,
    analyst_name,
    analysis_date,
)

# %% spatial plot of cd1 neck after gui track switching + boundary lines
sleapf.plot_spatial_switch(
    df_filled_switch_v4[df_filled_switch_v4["track"] == "cd1"],
    "Neck.x",
    "Neck.y",
    "neck cd1 after gui cleaning",
    x_bound_left_cd1,
    x_bound_right_cd1,
    y_bound_cd1,
    x_bound_left_bl6,
    x_bound_right_bl6,
    y_bound_bl6,
    folder_path,
    experimenter_name,
    mouse_id,
    session,
    experiment_id,
    analyst_name,
    analysis_date,
)

# %% save cleaned tracks as .csv file
file_name = f"{experimenter_name}_{mouse_id}_{session}_{experiment_id}_{analyst_name}_{analysis_date}_tracks_switched.csv"
save_path = os.path.join(folder_path, file_name)
df_filled_switch_v4.to_csv(save_path, index=False)

# %% read cleaned tracks in again
csv_path_cleaned = os.path.join(
    folder_path,
    f"{experimenter_name}_{mouse_id}_{session}_{experiment_id}_{analyst_name}_{analysis_date}_tracks_switched.csv",
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

# %%
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


# %% create empty tensor
mask_frames_bl6 = sleapf.empty_frame_tensor(video_path)

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

# %% #TODO in roi ang dir
# get x- and y- values of neck and nose of bl6 mouse
v_x_neck_bl6 = df_bl6["Neck.x"].values
v_y_neck_bl6 = df_bl6["Neck.y"].values
v_x_nose_bl6 = df_bl6["Nose.x"].values
v_y_nose_bl6 = df_bl6["Nose.y"].values

# get x- and y-value of neck cd1
v_x_neck_cd1 = df_cd1["Neck.x"].values
v_y_neck_cd1 = df_cd1["Neck.y"].values

# x- and y- coordinates of origin
x_origin = 0
y_origin = 0

# calculate origin vectors
o_neck_bl6 = np.array([v_x_neck_bl6 - x_origin, v_y_neck_bl6 - y_origin]).T
o_nose_bl6 = np.array([v_x_nose_bl6 - x_origin, v_y_nose_bl6 - y_origin]).T
o_neck_cd1 = np.array([v_x_neck_cd1 - x_origin, v_y_neck_cd1 - y_origin]).T

# calculate neck-nose and nose-mid_box vectors
vector_neck_bl6_nose_bl6 = o_nose_bl6 - o_neck_bl6
vector_neck_bl6_neck_cd1 = o_neck_bl6 - o_neck_cd1

# calculate the angle between the vectors
angles = sleapf.angle_between_vectors(
    vector_neck_bl6_nose_bl6, vector_neck_bl6_neck_cd1
)

# calculate the cross product for every frame
cross_products = np.cross(vector_neck_bl6_nose_bl6, vector_neck_bl6_neck_cd1)

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

# %%
