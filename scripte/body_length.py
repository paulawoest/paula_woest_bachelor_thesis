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
    for mouse_ID_path in Path(root_path).iterdir():
        if not mouse_ID_path.is_dir():
            continue
        if not mouse_ID_path.name.isdigit():
            continue
        for session_path in mouse_ID_path.iterdir():
            for csv_path in session_path.iterdir():
                if not csv_path.name.endswith("bl6_data_cleaned.csv"):
                    continue
                print(f"Loading csv file: {csv_path.name}")
                df = pd.read_csv(csv_path, index_col=0)
                df = df.assign(session=int(session_path.name[-1]))
                df = df.assign(mouse_ID=int(mouse_ID_path.name))
                data.append(df)
    return pd.concat(data, axis=0)


all_data = load_data(root_path)
# coords = all_data[["mouse_ID", "session", "Nose.x", "Nose.y", "Neck.x", "Neck.y", "Tail_Base.x", "Tail_Base.y"]]
# groups = coords.groupby(["mouse_ID", "session"])
# %%
data1 = all_data[all_data["session"] == 1]
data1_2283 = data1[data1["mouse_ID"]==2283]


# %%
def calculate_distances(row):
    # Extract coordinates
    nose_x, nose_y = row["Nose.x"], row["Nose.y"]
    tail_x, tail_y = row["Tail_Base.x"], row["Tail_Base.y"]

    # Calculate distances
    distance_nose_to_tail = np.sqrt((tail_x - nose_x) ** 2 + (tail_y - nose_y) ** 2)
    # distance_nose_to_origin = np.sqrt(nose_x**2 + nose_y**2)
    # distance_tail_to_origin = np.sqrt(tail_x**2 + tail_y**2)

    return distance_nose_to_tail  # , distance_nose_to_origin, distance_tail_to_origin


# Apply the function to your dataframe
# data1["Distances"] = data1.apply(calculate_distances, axis=1)
distance_nose_to_tail = data1.apply(calculate_distances, axis=1)
# %%
data1.loc[:, "distance_nose_to_tail"] = distance_nose_to_tail

# %%
colnames = ["distance_nose_to_tail"]
feat_names = [
    feat + "_" + name
    # fh = first half, sh = second half
    for feat in "mean".split()
    for name in colnames
]
feat_names = ["track", "mouse_ID", *feat_names]

# %%
feature_matrix_1 = []
for (track, mouse_ID), data in data1.groupby(["track", "mouse_ID"]):

    distance = data[["distance_nose_to_tail"]]

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
        *distance.iloc[:one_minute_int].mean(),
        # *distance.iloc[half_point:].mean(),
    ]

    feature_matrix_1.append(new_col)

feature_matrix_1 = pd.DataFrame(feature_matrix_1, columns=feat_names)


# %%
data2 = all_data[all_data["session"] == 2]


# %%
def calculate_distances_2(row):
    # Extract coordinates
    nose_x, nose_y = row["Nose.x"], row["Nose.y"]
    tail_x, tail_y = row["Tail_Base.x"], row["Tail_Base.y"]

    # Calculate distances
    distance_nose_to_tail = np.sqrt((tail_x - nose_x) ** 2 + (tail_y - nose_y) ** 2)
    # distance_nose_to_origin = np.sqrt(nose_x**2 + nose_y**2)
    # distance_tail_to_origin = np.sqrt(tail_x**2 + tail_y**2)

    return distance_nose_to_tail  # , distance_nose_to_origin, distance_tail_to_origin


# Apply the function to your dataframe
# data1["Distances"] = data1.apply(calculate_distances, axis=1)
distance_nose_to_tail_2 = data2.apply(calculate_distances_2, axis=1)

# %%
data2.loc[:, "distance_nose_to_tail"] = distance_nose_to_tail_2

# %%
colnames = ["distance_nose_to_tail"]
feat_names = [
    feat + "_" + name
    # fh = first half, sh = second half
    for feat in "mean".split()
    for name in colnames
]
feat_names = ["track", "mouse_ID", *feat_names]

# %%
feature_matrix_2 = []
for (track, mouse_ID), data in data2.groupby(["track", "mouse_ID"]):

    distance = data[["distance_nose_to_tail"]]

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
        *distance.iloc[:one_minute_int].mean(),
        # *distance.iloc[half_point:].mean(),
    ]

    feature_matrix_2.append(new_col)

feature_matrix_2 = pd.DataFrame(feature_matrix_2, columns=feat_names)


# %%
import matplotlib.pyplot as plt
import pandas as pd

# Load your dataset
# Replace 'your_dataset.csv' with the path to your data file
data = data1
# Assuming your variable with meaned values is a Pandas Series or another iterable


# Convert to a list
# Extract the column from the DataFrame and convert it to a list
meaned_values = data1['distance_nose_to_tail'].tolist()

# Create the histogram
plt.figure(figsize=(10, 6))
plt.hist(meaned_values, bins=100, color="#999999", edgecolor='black', alpha=0.7)
#for mean_value in meaned_values:
    #plt.axvline(mean_value, color='red', linestyle='-', linewidth=1.5, label=f'Mean: {mean_value}' if mean_value == meaned_values[0] else "")

# Customize the histogram
plt.xlabel('Distance (Nose to Tail)', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
#plt.xlim(50, 250)
# Show the plot
plt.grid(axis='y', linestyle='-', alpha=0.7)
plt.tight_layout()
plt.show()



# %%
