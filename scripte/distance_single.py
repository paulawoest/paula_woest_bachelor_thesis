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


# %% replace zeros in 'Neck.score' column with nan
df_bl6["Neck.score"] = df_bl6["Neck.score"].replace(0, np.nan)
df_bl6["Neck.score_scaled"] = df_bl6["Neck.score_scaled"].replace(0, np.nan)


# %% clean out scores below 20% percentile
# TODO change threshold if necessary
threshold_bl6 = np.nanpercentile(df_bl6["Neck.score"], 5)
# returns df_bl6 columns without track and frame_idx
columns_to_clean = df_bl6.columns.difference(["track", "frame_idx"])
df_bl6.loc[df_bl6["Neck.score"] <= threshold_bl6, columns_to_clean] = np.nan


# %% define coordinates columns
columns_coords = ["Nose.x", "Nose.y", "Neck.x", "Neck.y", "Tail_Base.x", "Tail_Base.y"]

# %% apply moving median filter to all coordinate columns
# TODO change if necessary
df_bl6 = sleapf.moving_median_filter(df_bl6, columns_coords, window=10)  # kernel size


# %%
df_bl6[columns_coords] = df_bl6[columns_coords].interpolate(method="linear", axis=0)
# backward fill to handle initial NaN values, if any
df_bl6[columns_coords] = df_bl6[columns_coords].bfill()

# %%
csv_file_path_cleaned = os.path.join(
    folder_path,
    f"{experimenter_name}_{mouse_id}_{session}_{experiment_id}_{analyst_name}_{analysis_date}_data_cleaned.csv",
)

# Save the DataFrame to CSV
df_bl6.to_csv(csv_file_path_cleaned)

# %%
df_video_info = pd.read_csv(os.path.join(os.getcwd(), "video_info.csv"))
df_video_info_coh = df_video_info[df_video_info["cohort"] == cohort_number]

# get coordinates for calculating the middle of the box
x_bound_left_cd1 = df_video_info_coh.x_bound_left_cd1.values[0]
x_bound_right_cd1 = df_video_info_coh.x_bound_right_cd1.values[0]
y_bound_bl6 = df_video_info_coh.y_bound_bl6.values[0]

midB = np.zeros((2), dtype=np.float64)
midB[0] = (x_bound_left_cd1 + x_bound_right_cd1) / 2  # x coordinates
midB[1] = y_bound_bl6

midB[1] = midB[1] * (2 / 3)


# %%
file_path = r"C:\Users\Cystein\Paula\distance_single.csv"

df_dist = pd.read_csv(file_path, sep=";")
# Calculate the distance between the different body parts of the mice

# df_dist = pd.DataFrame()
# df_dist['frame_idx'] = df_bl6['frame_idx'].copy()
df_dist[f"{mouse_id}"] = np.sqrt(
    (df_bl6["Nose.x"] - midB[0]) ** 2 + (df_bl6["Nose.y"] - midB[1]) ** 2
)

df_dist.to_csv(file_path, index=False, sep=";")

# %%
