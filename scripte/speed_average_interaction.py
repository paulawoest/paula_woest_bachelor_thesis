# %% importing libraries
import cv2
import numpy as np
import os
import pandas as pd
from csv import writer
from datetime import date

# import analysis functions
# import td_id_analysis_sleap_svenja_functions as sleapf

# %%
# directory where this script/file is saved
script_dir = os.path.dirname(
    os.path.abspath(r"C:\Users\Paula Woest\OneDrive\Desktop\Bachelore Arbeit")
)
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

# %%
# Definiere den Dateipfad
file_path = r"C:\Users\Paula Woest\OneDrive\Desktop\Bachelore Arbeit\SB_2283_1_uSoIn_PW_2024-10-29_data_cleaned.csv"

# Lade die Excel-Datei und wähle die entsprechende Spalte
# Angenommen, die Werte für `x_coords` stehen in einer Spalte namens "x_coords"
data = pd.read_csv(file_path)

# Speichere die Werte aus der Spalte "x_coords" in eine Variable
x_coords = data["Nose.x"].values  # `.values` konvertiert die Spalte in ein NumPy-Array
y_coords = data["Nose.y"].values
print("x_coords:", x_coords)
print("y_coords:", y_coords)


# %%
# x_coords = np.array([/* list of x-coordinates across frames */])
# y_coords = np.array([/* list of y-coordinates across frames */])
time_per_frame = 1 / 59.94  # assuming 59.94 frames per second

# Calculate distances between consecutive frames
distances = np.sqrt((np.diff(x_coords) ** 2) + (np.diff(y_coords) ** 2))

# Total distance traveled
total_distance = np.sum(distances)

# Total time (in seconds)
total_time = len(x_coords) * time_per_frame

# Average speed (distance per time)
average_speed = total_distance / total_time

print("Average speed of the mouse:", average_speed, "units per second")

# %%
import matplotlib.pyplot as plt
import numpy as np

# Extrahiere die x- und y-Koordinaten aus den entsprechenden Spalten
x_coords = data["Nose.x"].values
y_coords = data["Nose.y"].values
time_per_frame = 1 / 59.94  # zum Beispiel 59,94 FPS, daher 1/59,94 Sekunde pro Frame

# Berechne die Distanzen zwischen aufeinanderfolgenden Frames
distances_nose = np.sqrt((np.diff(x_coords) ** 2) + (np.diff(y_coords) ** 2))

# Berechne die Geschwindigkeit für jeden Frame-Übergang
speeds_nose = distances_nose / time_per_frame

# Berechne die durchschnittliche Geschwindigkeit
average_speed = np.mean(speeds_nose)
# Plotten der durchschnittlichen Geschwindigkeit als Linie
plt.figure(figsize=(10, 6))
plt.plot(speeds_nose, label="Instantaneous Speed", color="blue")
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
# %%
import matplotlib.pyplot as plt

# Example data: replace `x_coords` and `y_coords` with your actual data
# x_coords = [1, 2, 3, 4, 5, 6, 7, 8, 9]  # Replace with your x-coordinates
# y_coords = [1, 1.5, 2, 3, 2.5, 2, 1, 0.5, 0]  # Replace with your y-coordinates

# Plotting the spatial trajectory
plt.figure(figsize=(8, 8))
plt.plot(x_coords, y_coords, marker="o", color="blue", linestyle="-", markersize=4)

# Adding labels and title
plt.xlabel("X Coordinate (units)")
plt.ylabel("Y Coordinate (units)")
plt.title("Spatial Trajectory of Mouse Over Time")
plt.grid(True)

# Optional: add arrow to show direction
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
# %%
import numpy as np

# Example data: replace this with your actual x_coords data
# x_coords = np.array([1, 2, np.nan, 4, 5, np.nan, 7, 8, 9])  # Replace with your data

# Find indices where x_coords has NaN values
nan_indices = np.where(np.isnan(x_coords))[0]

# Print the indices of NaN values
print("Indices of NaN values in x_coords:", nan_indices)
print("Number of NaN values:", len(nan_indices))

# %%
import numpy as np
import matplotlib.pyplot as plt

# Example data: replace these with your actual x and y coordinates
# x_coords = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])  # Replace with actual x-coordinates
# y_coords = np.array([1, 1.5, 2, 3, 2.5, 2, 1, 0.5, 0])  # Replace with actual y-coordinates

# Calculate Euclidean distances between consecutive points
distances = np.sqrt(np.diff(x_coords) ** 2 + np.diff(y_coords) ** 2)

# Plotting the histogram of Euclidean distances
plt.figure(figsize=(8, 6))
plt.hist(distances, bins=100, color="skyblue", edgecolor="black")
plt.xlabel("Euclidean Distance (units)")
plt.ylabel("Frequency")
plt.title("Histogram of Euclidean Distances Between Consecutive Points")
plt.grid(True)
plt.ylim(0, 500)
plt.show()

# %%
import matplotlib.pyplot as plt
import numpy as np

# Extrahiere die x- und y-Koordinaten aus den entsprechenden Spalten
x_coords = data["Tail_Base.x"].values
y_coords = data["Tail_Base.y"].values
time_per_frame = 1 / 59.94  # zum Beispiel 59,94 FPS, daher 1/59,94 Sekunde pro Frame

# Berechne die Distanzen zwischen aufeinanderfolgenden Frames
distances_tail_base = np.sqrt((np.diff(x_coords) ** 2) + (np.diff(y_coords) ** 2))

# Berechne die Geschwindigkeit für jeden Frame-Übergang
speeds_tail_base = distances_tail_base / time_per_frame

# Berechne die durchschnittliche Geschwindigkeit
average_speed = np.mean(speeds_tail_base)
# Plotten der durchschnittlichen Geschwindigkeit als Linie
plt.figure(figsize=(10, 6))
plt.plot(speeds_tail_base, label="Instantaneous Speed", color="blue")
plt.axhline(
    y=average_speed,
    color="red",
    linestyle="--",
    label=f"Average Speed: {average_speed:.2f} units/s",
)
plt.title("Mouse Speed_tail_base per Frame with Average Speed")
plt.xlabel("Frame")
plt.ylabel("Speed (units per second)")
plt.legend()
plt.grid(True)
plt.show()
# %%
import numpy as np

# Example data: replace `distances` with your actual distance array
# distances = np.array([10, 15, 20, 18, 22, 25, 30])  # Replace with actual data

# Compute the difference vector along rows (differences between consecutive distances)
difference_vector = np.diff(distances_tail_base)

difference_vector_abs = np.absolute(difference_vector)
# Plotting the histogram of Euclidean distances
plt.figure(figsize=(8, 6))
# plt.hist(difference_vector_abs, bins=100, color="skyblue", edgecolor="black")
plt.hist(difference_vector_abs / 60, bins=100, color="skyblue", edgecolor="black")
plt.xlabel("Absolute distance (units)")
plt.ylabel("Frequency")
plt.title("Histogram of absolut differences")
plt.grid(True)
plt.ylim(0, 500)
plt.show()

# %%
# Calculate the 95th percentile
percentile_99 = np.percentile(difference_vector_abs, 99)

# Output the result
print("99th percentile:", percentile_99)

result = np.where(
    difference_vector_abs > percentile_99, np.nan, difference_vector_abs
)  # Set values greater than scalar to NaN
# Find indices where x_coords has NaN values
nan_indices = np.where(np.isnan(result))[0]

# Print the indices of NaN values
print("Indices of NaN values in x_coords:", nan_indices)
print("Number of NaN values:", len(nan_indices))

# %%
median = np.nanmedian(result)  # Median of the vector
q1 = np.nanpercentile(result, 25)  # 25th percentile (Q1)
q3 = np.nanpercentile(result, 75)  # 75th percentile (Q3)
iqr = q3 - q1  # Interquartile range

print(f"Median: {median}")
print(f"IQR: {iqr}")
# %%
import numpy as np

# Example data: replace `vector` with your actual vector and `scalar` with the threshold value
# vector = np.array([1, 5, 8, 3, 10, 7])  # Replace with your vector
scalar = 5  # Replace with your scalar value

# Create a logical vector that shows which elements are larger than the scalar
logical_vector = vector > scalar

# Output the result
print("Vector:", vector)
print("Logical vector (True if element > scalar):", logical_vector)

# %%
import numpy as np
import matplotlib.pyplot as plt

# Assuming `speeds` is your array of instantaneous speed data
window_size = 100  # Adjust this window as needed for smoothing
moving_avg_speed = np.convolve(
    speeds_tail_base, np.ones(window_size) / window_size, mode="valid"
)

# Plotting the moving average along with the instantaneous speed
plt.figure(figsize=(10, 6))
plt.plot(speeds_tail_base, color="blue", label="Instantaneous Speed")
plt.plot(
    moving_avg_speed,
    color="orange",
    linestyle="--",
    label=f"Moving Average Speed ({window_size}-frame window)",
)
plt.axhline(
    y=np.mean(speeds_tail_base),
    color="red",
    linestyle="--",
    label=f"Average Speed: {np.mean(speeds_tail_base):.2f} units/s",
)

plt.xlabel("Frame")
plt.ylabel("Speed (units per second)")
plt.title("Mouse Speed with Moving Average")
plt.legend()
plt.grid(True)
plt.show()

# %%
import matplotlib.pyplot as plt
import numpy as np

# Extrahiere die x- und y-Koordinaten aus den entsprechenden Spalten
x_coords = data["Neck.x"].values
y_coords = data["Neck.y"].values
time_per_frame = 1 / 59.94  # zum Beispiel 59,94 FPS, daher 1/59,94 Sekunde pro Frame

# Berechne die Distanzen zwischen aufeinanderfolgenden Frames
distances_neck = np.sqrt((np.diff(x_coords) ** 2) + (np.diff(y_coords) ** 2))

# Berechne die Geschwindigkeit für jeden Frame-Übergang
speeds_neck = distances_neck / time_per_frame

# Berechne die durchschnittliche Geschwindigkeit
average_speed = np.mean(speeds_neck)
# Plotten der durchschnittlichen Geschwindigkeit als Linie
plt.figure(figsize=(10, 6))
plt.plot(speeds_neck, label="Instantaneous Speed", color="blue")
plt.axhline(
    y=average_speed,
    color="red",
    linestyle="--",
    label=f"Average Speed: {average_speed:.2f} units/s",
)
plt.title("Mouse Speed_neck per Frame with Average Speed")
plt.xlabel("Frame")
plt.ylabel("Speed (units per second)")
plt.legend()
plt.grid(True)
plt.show()
# %%
import numpy as np

# Example data (replace these with your actual data)
speed_part1 = np.array(speeds_tail_base)
speed_part2 = np.array(speeds_nose)
speed_part3 = np.array(speeds_neck)

# Stack the vectors into a 2D array
all_speeds = np.array([speed_part1, speed_part2, speed_part3])

# Calculate the element-wise average
average_speed = np.mean(all_speeds, axis=0)

print("Average speed vector:", average_speed)

# %%
# Plotting
plt.figure(figsize=(10, 6))

# Plot each body part's speed vector
plt.plot(speed_part1, label="Speed Part 1", marker="o")
plt.plot(speed_part2, label="Speed Part 2", marker="o")
plt.plot(speed_part3, label="Speed Part 3", marker="o")

# Plot the average speed vector
plt.plot(
    average_speed, label="Average Speed", color="black", linestyle="--", marker="x"
)

# Add labels and title
plt.xlabel("Time Point")
plt.ylabel("Speed (units)")
plt.title("Speed of Body Parts with Average Speed")
plt.legend()
plt.grid(True)
plt.show()
# %%
# Plotting only the average speed
plt.figure(figsize=(10, 6))
plt.plot(
    average_speed, color="black", linestyle="--", marker="o", label="Average Speed"
)

# Add labels and title
plt.xlabel("Time Point")
plt.ylabel("Average Speed (units)")
plt.title("Average Speed of Body Parts")
plt.legend()
plt.grid(True)
plt.show()

# %%
import numpy as np

# Example data: replace `speeds` with your actual speed data
# This array should contain the speed at each time point
speed_part1 = np.array(speeds_tail_base)
# all_speeds = np.array([speed_part1, speed_part2, speed_part3])
# Calculate the maximum speed
max_speed = np.max(speed_part1)

print("The maximum speed reached by the mouse is:", max_speed, "units/s")


# %%
import matplotlib.pyplot as plt

# Plotting the speed data
plt.figure(figsize=(5, 3))
plt.plot(speed_part1, label="Speed", marker="o")
plt.axhline(
    y=max_speed,
    color="red",
    linestyle="--",
    label=f"Max Speed: {max_speed:.2f} units/s",
)

# Mark the point where maximum speed is reached
max_index = np.argmax(speed_part1)
plt.plot(max_index, max_speed, "ro")  # red dot at max point

# Add labels and title
plt.xlabel("Time Point")
plt.ylabel("Speed (units/s)")
plt.title("Mouse Speed with Maximum Speed Highlighted")
plt.legend()
plt.grid(True)
plt.show()

# %%
import matplotlib.pyplot as plt

# Plotting the speed data
plt.figure(figsize=(10, 6))
plt.plot(speed_part1, label="Speed", marker="o")
plt.axhline(
    y=max_speed,
    color="red",
    linestyle="--",
    label=f"Max Speed: {max_speed:.2f} units/s",
)

# Mark the point where maximum speed is reached
max_index = np.argmax(speed_part1)
plt.plot(max_index, max_speed, "ro")  # red dot at max point

# Add labels and title
plt.xlabel("Time Point")
plt.ylabel("Speed (units/s)")
plt.title("Mouse Speed with Maximum Speed Highlighted")
plt.legend()
plt.grid(True)
plt.show()

# %%

# %%
import pandas as pd

# Definiere den Dateipfad
file_path = r"C:\Users\Paula Woest\OneDrive\Desktop\Bachelore Arbeit\SB_2283_1_uSoIn_PW_2024-10-29_data_cleaned.csv"

# Lade die Excel-Datei und wähle die entsprechende Spalte
# Angenommen, die Werte für `x_coords` stehen in einer Spalte namens "x_coords"
data = pd.read_csv(file_path)

# Speichere die Werte aus der Spalte "x_coords" in eine Variable
x_coords = data["Neck.x"].values  # `.values` konvertiert die Spalte in ein NumPy-Array
y_coords = data["Neck.y"].values
print("x_coords:", x_coords)
print("y_coords:", y_coords)
# %%
# Calculate the average location
mean_x = np.mean(x_coords)
mean_y = np.mean(y_coords)

average_location = (mean_x, mean_y)

print("Average location of the mouse:", average_location)

# %%
# %%
import matplotlib.pyplot as plt

# Example data: replace `x_coords` and `y_coords` with your actual data
x_coords = mean_x  # Replace with your x-coordinates
y_coords = mean_y  # Replace with your y-coordinates

# Plotting the spatial trajectory
plt.figure(figsize=(8, 8))
plt.plot(x_coords, y_coords, marker="o", color="blue", linestyle="-", markersize=4)

# Adding labels and title
plt.xlabel("X Coordinate (units)")
plt.ylabel("Y Coordinate (units)")
plt.title("Spatial Trajectory of Mouse Over Time")
plt.grid(True)

# Optional: add arrow to show direction
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

# %%
import numpy as np

# Example data: replace `distances` with your actual distance array
# distances = np.array([10, 15, 20, 18, 22, 25, 30])  # Replace with actual data

# Compute the difference vector along rows (differences between consecutive distances)
difference_vector = np.diff(distances_tail_base)

difference_vector_abs = np.absolute(difference_vector)
# Plotting the histogram of Euclidean distances
plt.figure(figsize=(8, 6))
# plt.hist(difference_vector_abs, bins=100, color="skyblue", edgecolor="black")
plt.hist(difference_vector_abs / 60, bins=100, color="skyblue", edgecolor="black")
plt.xlabel("Absolute distance (units)")
plt.ylabel("Frequency")
plt.title("Histogram of absolut differences")
plt.grid(True)
plt.ylim(0, 500)
plt.show()
# %%
# Calculate the 95th percentile
percentile_99 = np.percentile(difference_vector_abs, 99)

# Output the result
print("99th percentile:", percentile_99)

result = np.where(
    difference_vector_abs > percentile_99, np.nan, difference_vector_abs
)  # Set values greater than scalar to NaN
# Find indices where x_coords has NaN values
nan_indices = np.where(np.isnan(result))[0]

# Print the indices of NaN values
print("Indices of NaN values in x_coords:", nan_indices)
print("Number of NaN values:", len(nan_indices))

# %%
median = np.nanmedian(result)  # Median of the vector
q1 = np.nanpercentile(result, 25)  # 25th percentile (Q1)
q3 = np.nanpercentile(result, 75)  # 75th percentile (Q3)
iqr = q3 - q1  # Interquartile range

print(f"Median: {median}")
print(f"IQR: {iqr}")

# %%
import math


# direkte distanz zwischen den beiden Punkten Neck und Tail_base
def calculate_mouse_length(tail_base, neck):
    x1, y1 = tail_base
    x2, y2 = neck
    length = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return length


# Beispielkoordinaten für den Schwanzansatz und den Hals
tail_base = (2, 3)
neck = (7, 10)

# Berechnung der Länge
length = calculate_mouse_length(tail_base, neck)
print(f"Die Länge der Maus beträgt: {length} Einheiten")
