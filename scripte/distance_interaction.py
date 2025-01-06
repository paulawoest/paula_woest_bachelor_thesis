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
    folder_path,
    f"{experimenter_name}_{mouse_id}_{session}_{experiment_id}_{analyst_name}_{analysis_date}_tracks_switched.csv",
)
# %% save cleaned tracks as .csv file
file_name = f"{experimenter_name}_{mouse_id}_{session}_{experiment_id}_{analyst_name}_{analysis_date}_tracks_switched.csv"
save_path = os.path.join(folder_path, file_name)


# %% read cleaned tracks in again
csv_path_cleaned = os.path.join(
    folder_path,
    f"{experimenter_name}_{mouse_id}_{session}_{experiment_id}_{analyst_name}_2024-10-21_tracks_switched.csv",
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


# %% replace zeros in 'Neck.score' column with nan
df_bl6["Neck.score"] = df_bl6["Neck.score"].replace(0, np.nan)
df_bl6["Neck.score_scaled"] = df_bl6["Neck.score_scaled"].replace(0, np.nan)

df_cd1["Neck.score"] = df_cd1["Neck.score"].replace(0, np.nan)
df_cd1["Neck.score_scaled"] = df_cd1["Neck.score_scaled"].replace(0, np.nan)


# %% clean out scores below 20% percentile
# TODO change threshold if necessary
threshold_bl6 = np.nanpercentile(df_bl6["Neck.score"], 5)
threshold_cd1 = np.nanpercentile(df_cd1["Neck.score"], 5)
df_bl6, df_cd1 = sleapf.clean_data(threshold_bl6, threshold_cd1, df_bl6, df_cd1)


# %% define coordinates columns
columns_coords = ["Nose.x", "Nose.y", "Neck.x", "Neck.y", "Tail_Base.x", "Tail_Base.y"]

# %% apply moving median filter to all coordinate columns
# TODO change if necessary
df_bl6 = sleapf.moving_median_filter(df_bl6, columns_coords, window=10)  # kernel size
df_cd1 = sleapf.moving_median_filter(df_cd1, columns_coords, window=10)


# %%
df_bl6[columns_coords] = df_bl6[columns_coords].interpolate(method="linear", axis=0)
df_cd1[columns_coords] = df_cd1[columns_coords].interpolate(method="linear", axis=0)

# backward fill to handle initial NaN values, if any
df_bl6[columns_coords] = df_bl6[columns_coords].bfill()
df_cd1[columns_coords] = df_cd1[columns_coords].bfill()

# %%
bl6_csv_file_path_cleaned = os.path.join(
    folder_path,
    f"{experimenter_name}_{mouse_id}_{session}_{experiment_id}_{analyst_name}_{analysis_date}_bl6_data_cleaned.csv",
)

# Save the DataFrame to CSV
df_bl6.to_csv(bl6_csv_file_path_cleaned)

cd1_csv_file_path_cleaned = os.path.join(
    folder_path,
    f"{experimenter_name}_{mouse_id}_{session}_{experiment_id}_{analyst_name}_{analysis_date}_cd1_data_cleaned.csv",
)

# Save the DataFrame to CSV
df_cd1.to_csv(cd1_csv_file_path_cleaned)
# %%
file_path = r"C:\Users\Cystein\Paula\distance_interaction.csv"

df_dist = pd.read_csv(file_path, sep=";")
# Calculate the distance between the Ear and Nose for each frame

# df_dist = pd.DataFrame()
# df_dist['frame_idx'] = df_bl6['frame_idx'].copy()
df_dist[f"{mouse_id}"] = np.sqrt(
    (df_bl6["Nose.x"] - df_cd1["Nose.x"]) ** 2
    + (df_bl6["Nose.y"] - df_cd1["Nose.y"]) ** 2
)
# %%
df_dist.to_csv(file_path, index=False, sep=";")

# %%
import numpy as np


def calculate_hypotenuse(a, b):
    return np.sqrt(a**2 + b**2)


# Example usage:
side_a = 473.227935791015 - 683.969116210937
side_b = 237.023956298828 - 175.583465576171
hypotenuse = calculate_hypotenuse(side_a, side_b)
print(f"The length of the hypotenuse is: {hypotenuse}")

# %%
