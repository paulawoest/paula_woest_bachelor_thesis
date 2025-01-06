# %% Imports
import pandas as pd
import os
import shutil

# %% Load the Excel file
# change path accordingly
csv_file = r"C:\Users\Cystein\Maja\trial_number_corresponding_mouse_id.csv"
df = pd.read_csv(csv_file)

# %% Rename Videos
# Iterate over each row in the DataFrame
for index, row in df.iterrows():
    # Generate the full filename
    original_full_filename = f"{row['cohort']}_{row['trial_number']}.mp4"

    # Get the new ID
    new_id = row["mouse_id"]

    # generate new full filename
    new_full_filename = f"{row['cohort']}_{new_id}_session{row['session']}.mp4"

    # Rename the file
    if os.path.exists(original_full_filename):
        os.rename(original_full_filename, new_full_filename)
        print(f"Renamed {original_full_filename} to {new_full_filename}.mp4")
    else:
        print(f"File {original_full_filename} does not exist.")