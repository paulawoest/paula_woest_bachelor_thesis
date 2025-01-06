# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

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
data1 = all_data[all_data["session"] == 1]
data2 = all_data[all_data["session"] == 2]
# %%
time_per_frame = 1 / 59.94


# %%
def average_speed(row, prev_row, time_per_frame):
    # Extract current and previous coordinates
    neck_x, neck_y = row["Neck.x"], row["Neck.y"]
    prev_neck_x, prev_neck_y = prev_row["Neck.x"], prev_row["Neck.y"]
    # Calculate distance
    distance = np.sqrt((neck_x - prev_neck_x) ** 2 + (neck_y - prev_neck_y) ** 2)
    # Calculate speed
    speed = distance / time_per_frame
    return speed, distance


# %%time_per_frame = 1  # Example time per frame

# Compute speed manually
speeds = [0]  # Speed for the first row is zero because there's no previous row
distances = [0]
for i in range(1, len(data1)):
    current_row = data1.iloc[i]
    prev_row = data1.iloc[i - 1]
    speed, distance = average_speed(current_row, prev_row, time_per_frame)
    speeds.append(speed)
    distances.append(distance)


# %%
def vector_abs_differences(distance):
    difference_vector = np.diff(distance)
    difference_vector = np.insert(difference_vector, 0, 0)
    difference_vector_abs = np.absolute(difference_vector)
    plt.figure(figsize=(8, 6))
    plt.hist(difference_vector_abs / 60, bins=100, color="skyblue", edgecolor="black")
    plt.xlabel("Absolute distance (units)")
    plt.ylabel("Frequency")
    plt.title("Histogram of absolut differences")
    plt.grid(True)
    plt.ylim(0, 500)
    plt.show()

    return difference_vector_abs


# %%
def percentile_99_speed(difference_vector_abs):
    percentile_99 = np.percentile(difference_vector_abs, 99)
    result = np.where(
        difference_vector_abs > percentile_99, np.nan, difference_vector_abs
    )
    # nan_indices = np.where(np.isnan(result))[0]

    return result


# %%
def median_iqr_speed(result):
    """
    Calculate median and interquartile range.
    """
    median = np.nanmedian(result)  # Median of the vector
    q1 = np.nanpercentile(result, 25)  # 25th percentile (Q1)
    q3 = np.nanpercentile(result, 75)  # 75th percentile (Q3)
    iqr = q3 - q1

    # print(f"Median: {median}")
    # print(f"IQR: {iqr}")

    return median, iqr


# %%
data1.loc[:, "distance"] = distances
data1.loc[:, "speed"] = speeds

# %%
difference_vector_abs = vector_abs_differences(distances)
result = percentile_99_speed(difference_vector_abs)

# %%
data1.loc[:, "result"] = result

# %%
colnames = ["mean_distance", "median_speed", "iqr_speed"]
feat_names = [
    name
    # fh = first half, sh = second half
    # for feat in "mean".split()
    for name in colnames
]
feat_names = ["track", "mouse_ID", *feat_names]

# %%
feature_matrix_1 = []
# feature_matrix_2 = []
for (track, mouse_ID), data in data1.groupby(["track", "mouse_ID"]):

    distance = data[["distance"]]
    result = data[["result"]]

    # Step 1: Count the number of frames
    num_frames = len(distance)

    # Step 2: Divide by two to find the midpoint
    one_minute = num_frames // 2.53  # Integer division
    one_minute_int = round(one_minute)  # Integer division

    """
    mean_loc_first_half = coordinates.iloc[:half_point].mean()
    mean_loc_second_half = coordinates.iloc[half_point:].mean()
    median_loc_first_half = coordinates.iloc[:half_point].median()
    median_loc_second_half = coordinates.iloc[half_point:].median()
    std_loc_first_half = coordinates.iloc[:half_point].std()
    std_loc_second_half = coordinates.iloc[half_point:].std()
    """

    median, iqr = median_iqr_speed(result.iloc[:one_minute_int])
    # median_second_half, iqr_second_half = median_iqr_speed(result.iloc[half_point:])

    new_col = [
        track,
        mouse_ID,
        *distance.iloc[:one_minute_int].mean(),
        median,
        iqr,
    ]

    feature_matrix_1.append(new_col)

feature_matrix_1 = pd.DataFrame(feature_matrix_1, columns=feat_names)
# %%
data2 = data2.iloc[: len(distances)]  # Or len(speeds), depending on your data

# Now, you can safely assign distances and speeds:
data2.loc[:, "distance"] = distances
data2.loc[:, "speed"] = speeds


# %%
difference_vector_abs = vector_abs_differences(distances)
result = percentile_99_speed(difference_vector_abs)

# %%
data2.loc[:, "result"] = result

# %%
colnames = ["mean_distance", "median_speed", "iqr_speed"]
feat_names = [
    name
    # fh = first half, sh = second half
    # for feat in "mean".split()
    for name in colnames
]
feat_names = ["track", "mouse_ID", *feat_names]

# %%
feature_matrix_2 = []
# feature_matrix_2 = []
for (track, mouse_ID), data in data2.groupby(["track", "mouse_ID"]):

    distance = data[["distance"]]
    result = data[["result"]]

    # Step 1: Count the number of frames
    num_frames = len(distance)

    # Step 2: Divide by two to find the midpoint
    one_minute = num_frames // 2.53  # Integer division
    one_minute_int = round(one_minute)  # Integer division

    """
    mean_loc_first_half = coordinates.iloc[:half_point].mean()
    mean_loc_second_half = coordinates.iloc[half_point:].mean()
    median_loc_first_half = coordinates.iloc[:half_point].median()
    median_loc_second_half = coordinates.iloc[half_point:].median()
    std_loc_first_half = coordinates.iloc[:half_point].std()
    std_loc_second_half = coordinates.iloc[half_point:].std()
    """

    median, iqr = median_iqr_speed(result.iloc[:one_minute_int])
    # median_second_half, iqr_second_half = median_iqr_speed(result.iloc[half_point:])

    new_col = [
        track,
        mouse_ID,
        *distance.iloc[:one_minute_int].mean(),
        median,
        iqr,
    ]

    feature_matrix_2.append(new_col)

feature_matrix_2 = pd.DataFrame(feature_matrix_2, columns=feat_names)
# %%
import os

folder_path_1 = os.path.join(root_path, "output_data", "1min_speed_iqr_median_1.csv")

folder_path_2 = os.path.join(root_path, "output_data", "1min_speed_iqr_median_2.csv")

# Save in_roi_ang_dir dataframe as .csv file
feature_matrix_1.to_csv(folder_path_1, index=False)
feature_matrix_2.to_csv(folder_path_2, index=False)
# %%
