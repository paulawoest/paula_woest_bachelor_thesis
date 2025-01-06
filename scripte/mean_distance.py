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
def calculate_distances_1(row, midB):
    # Extract coordinates
    neck_x, neck_y = row["Neck.x"], row["Neck.y"]
    midB_x = midB[0]
    midB_y = midB[1]

    # Calculate distances
    distance_neck_box = np.sqrt((midB_x - neck_x) ** 2 + (midB_y - neck_y) ** 2)

    return distance_neck_box


# %%
data1 = all_data[all_data["session"] == 1]
data6 = data1[data1["mouse_ID"] <= 2301]  # change the numbers of the different
data7 = data1[data1["mouse_ID"] > 2301]
distance6 = data6.apply(calculate_distances_1, axis=1, args=(midB_6,))
distance7 = data7.apply(calculate_distances_1, axis=1, args=(midB_7,))
data6.loc[:, "distance"] = distance6
data7.loc[:, "distance"] = distance7

# %%
colnames = ["distance"]
feat_names = [
    feat + "_" + name  # fh = first half, sh = second half
    for feat in "mean".split()
    for name in colnames
]
feat_names = ["track", "mouse_ID", *feat_names]

# %%
feature_matrix_1_6 = []
for (track, mouse_ID), data in data6.groupby(["track", "mouse_ID"]):

    distance = data[["distance"]]

    # Step 1: Count the number of frames
    num_frames = len(distance)

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
        # *distance.mean(),
        *distance.iloc[:one_minute_int].mean(),
    ]

    feature_matrix_1_6.append(new_col)

feature_matrix_1_6 = pd.DataFrame(feature_matrix_1_6, columns=feat_names)


# %%
feature_matrix_1_7 = []
for (track, mouse_ID), data in data7.groupby(["track", "mouse_ID"]):

    distance = data[["distance"]]

    # Step 1: Count the number of frames
    num_frames = len(distance)

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
        # *distance.mean(),
        *distance.iloc[:one_minute_int].mean(),
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
def calculate_distances_2(row):
    # Extract coordinates
    neck_x_bl6, neck_y_bl6 = row["Neck.x"], row["Neck.y"]
    neck_x_cd1, neck_y_cd1 = row["Neck_cd1.x"], row["Neck_cd1.y"]

    # Calculate distances
    distance_neck_box = np.sqrt(
        (neck_x_cd1 - neck_x_bl6) ** 2 + (neck_y_cd1 - neck_y_bl6) ** 2
    )

    return distance_neck_box


# %%
distance_2 = test.apply(calculate_distances_2, axis=1)

# %%
test.loc[:, "distance"] = distance_2

# %%
colnames = ["distance"]
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

    distance = data[["distance"]]

    # Step 1: Count the number of frames
    num_frames = len(distance)

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
        # *distance.mean(),
        *distance.iloc[:one_minute_int].mean(),
    ]

    feature_matrix_2.append(new_col)

feature_matrix_2 = pd.DataFrame(feature_matrix_2, columns=feat_names)

# %%
# %%
import os

folder_path_1 = os.path.join(root_path, "output_data", "1min_mean_distance_1.csv")

folder_path_2 = os.path.join(root_path, "output_data", "1min_mean_distance_2.csv")

# Save in_roi_ang_dir dataframe as .csv file
feature_matrix_1.to_csv(folder_path_1, index=False)
feature_matrix_2.to_csv(folder_path_2, index=False)
# %%
