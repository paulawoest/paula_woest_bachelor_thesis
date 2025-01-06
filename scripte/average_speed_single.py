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
bodypart_x = "Nose.x"
bodypart_y = "Nose.y"
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
def load_data(root_path):
    """
    Load all cleand data into one dataframe.
    """
    data = []
    for subject_path in Path(root_path).iterdir():
        if not subject_path.is_dir():
            continue
        if not subject_path.name.isdigit():
            continue
        for session_path in subject_path.iterdir():
            for csv_path in session_path.iterdir():
                if csv_path.name.endswith("data_cleaned.csv"):
                    print(f"Loading csv file: {csv_path.name}")
                    df = pd.read_csv(csv_path, index_col=0)
                    df = df.assign(session=int(session_path.name[-1]))
                    df = df.assign(subject=int(subject_path.name))
                    data.append(df)
    return pd.concat(data, axis=0)


data = load_data(root_path)
# %%
new_data = data[data["subject"] == 2283]
