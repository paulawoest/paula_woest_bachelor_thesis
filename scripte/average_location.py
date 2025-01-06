# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import os

# %%
# Definiere den Dateipfad
root_path = r"C:\Users\Paula Woest\OneDrive\Desktop\Bachelore Arbeit\Paula"


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
                if not csv_path.name.endswith("bl6_data_cleaned.csv"):
                    continue
                print(f"Loading csv file: {csv_path.name}")
                df = pd.read_csv(csv_path, index_col=0)
                df = df.assign(session=int(session_path.name[-1]))
                df = df.assign(subject=int(subject_path.name))
                data.append(df)
    return pd.concat(data, axis=0)


all_data = load_data(root_path)
# coords = all_data[["subject", "session", "Nose.x", "Nose.y", "Neck.x", "Neck.y", "Tail_Base.x", "Tail_Base.y"]]
# groups = coords.groupby(["subject", "session"])


# %%

colnames = ["Neck.x", "Neck.y"]
feat_names = [
    feat + "_location_" + part + "_" + name
    for part in ["fh", "sh"]  # fh = first half, sh = second half
    for feat in "mean".split()
    for name in colnames
]
feat_names = ["track", "subject", *feat_names]

feature_matrix_1 = []
feature_matrix_2 = []
for (track, subject, session), data in all_data.groupby(
    ["track", "subject", "session"]
):

    coordinates = data[["Neck.x", "Neck.y"]]

    # Step 1: Count the number of frames
    num_frames = len(coordinates)

    # Step 2: Divide by two to find the midpoint
    half_point = num_frames // 2  # Integer division

    """
    mean_loc_first_half = coordinates.iloc[:half_point].mean()
    mean_loc_second_half = coordinates.iloc[half_point:].mean()
    median_loc_first_half = coordinates.iloc[:half_point].median()
    median_loc_second_half = coordinates.iloc[half_point:].median()
    std_loc_first_half = coordinates.iloc[:half_point].std()
    std_loc_second_half = coordinates.iloc[half_point:].std()
    """

    new_col = [
        track,
        subject,
        *coordinates.iloc[:half_point].mean(),
        *coordinates.iloc[half_point:].mean(),
    ]

    if session == 1:
        feature_matrix_1.append(new_col)
    else:
        if track == "bl6":
            feature_matrix_2.append(new_col)

feature_matrix_1 = pd.DataFrame(feature_matrix_1, columns=feat_names)
feature_matrix_2 = pd.DataFrame(feature_matrix_2, columns=feat_names)

# %%
folder_path_1 = os.path.join(root_path, "output_data", "mean_location_session1.csv")

folder_path_2 = os.path.join(root_path, "output_data", "mean_location_session2.csv")

# Save in_roi_ang_dir dataframe as .csv file
feature_matrix_1.to_csv(folder_path_1, index=False)
feature_matrix_2.to_csv(folder_path_2, index=False)


# %%
# for root, dirs, files in os.walk(root_path):
#     if os.path.basename(root) == "session1":
#         for file in files:
#             if file.endswith("data_cleaned.csv"):
#                 old_file_path = os.path.join(root, file)
#                 if "data_cleaned" in file:
#                     new_file_name = file.replace("data_cleaned", "bl6_data_cleaned")
#                     new_file_path = os.path.join(root, new_file_name)

#                 try:
#                     df = pd.read_csv(old_file_path)
#                     if "Unnamed: 0" in df.columns:
#                         df = df.drop(columns=["Unnamed: 0"])
#                     df["track"] = "bl6"
#                     df.to_csv(new_file_path)
#                     print(f"modified and renamed {old_file_path} to {new_file_path}")
#                 except Exception as e:
#                     print(f"failed to modify {old_file_path}: {e}")
# %%
