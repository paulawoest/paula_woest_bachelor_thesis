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
    for subject_path in Path(root_path).iterdir():
        if not subject_path.is_dir():
            continue
        if not subject_path.name.isdigit():
            continue
        for session_path in subject_path.iterdir():
            for csv_path in session_path.iterdir():
                if not csv_path.name.endswith("data_cleaned.csv"):
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
def angle_between_vectors_cross(neck_x, neck_y, nose_x, nose_y, ref_x, ref_y):

    u = [(nose_x - neck_x), (nose_y - neck_y)]
    v = [(ref_x - neck_x), (ref_y - neck_y)]

    u = np.array(u)
    v = np.array(v)

    cross_product = np.cross(u, v)
    magnitude_u = np.linalg.norm(u)
    magnitude_v = np.linalg.norm(v)
    magnitude_cross = np.linalg.norm(cross_product)

    sin_theta = magnitude_cross / (magnitude_u * magnitude_v)
    sin_theta = np.clip(
        sin_theta, -1.0, 1.0
    )  # Ensure sin_theta is within the valid range

    angle_radians = np.arcsin(sin_theta)
    angle_degrees = np.degrees(angle_radians)

    return angle_degrees
