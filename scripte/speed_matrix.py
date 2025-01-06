# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pathlib import Path

# %%
# Definiere den Dateipfad
root_path = r"C:\Users\Paula Woest\OneDrive\Desktop\Bachelore Arbeit\Paula"

# Lade die Excel-Datei und wähle die entsprechende Spalte
# Angenommen, die Werte für `x_coords` stehen in einer Spalte namens "x_coords"
bodypart_x = "Neck.x"
bodypart_y = "Neck.y"
time_per_frame = 1 / 59.94


# %%
def average_speed_bodypart(data, bodypart_x, bodypart_y, time_per_frame):
    x_coords = data[
        bodypart_x
    ].values  # `.values` konvertiert die Spalte in ein NumPy-Array
    y_coords = data[bodypart_y].values

    distances = np.sqrt((np.diff(x_coords) ** 2) + (np.diff(y_coords) ** 2))
    speeds = distances / time_per_frame

    average_speed = np.mean(speeds)

    return speeds, average_speed, distances


# %%
def plot_average_speed(speeds, average_speed):
    # Plotten der durchschnittlichen Geschwindigkeit als Linie
    plt.figure(figsize=(10, 6))
    plt.plot(speeds, label="Instantaneous Speed", color="blue")
    plt.axhline(
        y=average_speed,
        color="red",
        linestyle="--",
        label=f"Average Speed: {average_speed:.2f} units/s",
    )
    plt.title("Mouse Speed_nose per Frame with Average Speed")
    plt.xlabel("Frame")
    plt.ylabel("Speed (units per second)")
    plt.legend()

    plt.grid(True)
    plt.show()


# %%
def plot_spatial_trajectory(x_coords, y_coords):
    plt.figure(figsize=(8, 8))
    plt.plot(x_coords, y_coords, marker="o", color="blue", linestyle="-", markersize=4)

    # Adding labels and title
    plt.xlabel("X Coordinate (units)")
    plt.ylabel("Y Coordinate (units)")
    plt.title("Spatial Trajectory of Mouse Over Time")
    plt.grid(True)

    for i in range(1, len(x_coords)):
        plt.arrow(
            x_coords[i - 1],
            y_coords[i - 1],
            x_coords[i] - x_coords[i - 1],
            y_coords[i] - y_coords[i - 1],
            head_width=0.1,
            head_length=0.1,
            color="blue",
            alpha=0.6,
        )
    plt.show()


def vector_abs_differences(distances):
    difference_vector = np.diff(distances)
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
time_per_frame = 1 / 59.94
# %%
import numpy as np


def average_speed(row, prev_row, time_per_frame):
    """
    Calculate the distance per frame for a specific body part and the average speed.

    Parameters:
    - row: Current row of the DataFrame (pandas Series).
    - prev_row: Previous row of the DataFrame (pandas Series).
    - time_per_frame: Time elapsed per frame (e.g., in seconds).
    Returns:
    - Speed: Average speed for the given body part between two frames.
    """
    # Extract current and previous coordinates
    neck_x, neck_y = row["Neck.x"], row["Neck.y"]
    prev_neck_x, prev_neck_y = prev_row["Neck.x"], prev_row["Neck.y"]

    # Calculate distance between frames
    distance = np.sqrt((neck_x - prev_neck_x) ** 2 + (neck_y - prev_neck_y) ** 2)

    # Calculate speed (distance / time)
    speed = distance / time_per_frame

    return speed, distance


# %%
def vector_abs_differences(distance):
    difference_vector = np.diff(distance)
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

    print(f"Median: {median}")
    print(f"IQR: {iqr}")


# %%
data1 = all_data[all_data["session"] == 1]
# %%
speed = data1.apply(average_speed(), axis=1)
# %%
