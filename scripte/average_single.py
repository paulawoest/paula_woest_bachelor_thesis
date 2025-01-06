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
mouse_id = int(input("Enter mouse_id number: "))
cohort_number = int(input("Enter cohort number: "))  # i.e. either 6 or 7

# %%
# Define the folder path
folder_path = r"C:\Users\Cystein\Paula"  # Replace with your actual folder path

# Construct the CSV file path
csv_path = r"C:\Users\Cystein\Paula\distance_interaction.csv"

# Read the CSV file into a DataFrame
df = pd.read_csv(csv_path, sep=";")
# Specify the column you want to average (replace 'ColumnName' with your actual column name)
column_name = str(mouse_id)  # Replace with your actual column name

# Define the chunk size
chunk_size = 250

# %%
# Calculate the averages in chunks
averages = []
for i in range(0, len(df), chunk_size):
    chunk = df[column_name].iloc[i : i + chunk_size]  # Select the current chunk
    averages.append(chunk.mean())  # Calculate and store the average

# Print the results
for index, avg in enumerate(averages):
    print(
        f"Average of values {index * chunk_size + 1} to {(index + 1) * chunk_size}: {avg}"
    )

# Create a DataFrame to hold the averages
df_ave = pd.DataFrame(
    {
        "Chunk Start": [i * chunk_size + 1 for i in range(len(averages))],
        "Chunk End": [(i + 1) * chunk_size for i in range(len(averages))],
        column_name: averages,
    }
)

# Save the averages DataFrame to a new Excel file
# averages_df.to_excel(output_file_path, index=False)

# %%
file_path = r"C:\Users\Cystein\Paula\average_interaction.csv"
df_ave_all = pd.read_csv(file_path, sep=";")
# Calculate the distance between the different body parts of the mice

# df_dist = pd.DataFrame()
# df_dist['frame_idx'] = df_bl6['frame_idx'].copy()
df_ave_all[column_name] = averages
df_ave_all.to_csv(file_path, index=False, sep=";")

# %%
