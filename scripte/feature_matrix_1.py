# %%
import pandas as pd
import os

# Read both CSV files
file1 = pd.read_csv(
    r"C:\Users\Paula Woest\OneDrive\Desktop\Bachelore Arbeit\Paula\output_data\Session 1_Single\1min_mean_location_session1.csv"
)  # Contains first set of features
file2 = pd.read_csv(
    r"C:\Users\Paula Woest\OneDrive\Desktop\Bachelore Arbeit\Paula\output_data\Session 1_Single\1min_mean_angle_1.csv"
)
file3 = pd.read_csv(
    r"C:\Users\Paula Woest\OneDrive\Desktop\Bachelore Arbeit\Paula\output_data\Session 1_Single\1min_mean_length_1.csv"
)
file4 = pd.read_csv(
    r"C:\Users\Paula Woest\OneDrive\Desktop\Bachelore Arbeit\Paula\output_data\Session 1_Single\1min_mean_distance_1.csv"
)
file5 = pd.read_csv(
    r"C:\Users\Paula Woest\OneDrive\Desktop\Bachelore Arbeit\Paula\output_data\Session 1_Single\1min_speed_iqr_median_1.csv"
)
file6 = pd.read_csv(
    r"C:\Users\Paula Woest\OneDrive\Desktop\Bachelore Arbeit\Paula\output_data\Session 1_Single\seconds_info_1.csv"
)
file7 = pd.read_csv(
    r"C:\Users\Paula Woest\OneDrive\Desktop\Bachelore Arbeit\Paula\output_data\sus_res_data.csv"
)
# Contains second set of features
# Specify columns to keep from each file
features1 = [
    "mouse_ID",
    "mean_location__Neck.x",
    "mean_location__Neck.y",
]
features2 = [
    "mouse_ID",
    "mean_angle",
]
features3 = [
    "mouse_ID",
    "mean_distance_nose_to_tail",
]
features4 = [
    "mouse_ID",
    "mean_distance",
]
features5 = [
    "mouse_ID",
    "median_speed",
    "iqr_speed",
]
features6 = [
    "mouse_ID",
    "s_in_roi",
    "s_in_roi_ang_dir",
]
features7 = [
    "mouse_ID",
    "sus_res",
]
# Replace with actual column names from second file
# Select relevant columns
df1 = file1[features1]
df2 = file2[features2]
df3 = file3[features3]
df4 = file4[features4]
df5 = file5[features5]
df6 = file6[features6]
df7 = file7[features7]
# %%
# Merge datasets on mouse_ID
combined_features = pd.merge(df1, df2, on="mouse_ID", how="outer")
combined_features_2 = pd.merge(combined_features, df3, on="mouse_ID", how="outer")
combined_features_3 = pd.merge(combined_features_2, df4, on="mouse_ID", how="outer")
combined_features_4 = pd.merge(combined_features_3, df5, on="mouse_ID", how="outer")
combined_features_5 = pd.merge(combined_features_4, df6, on="mouse_ID", how="outer")
combined_features_6 = pd.merge(combined_features_5, df7, on="mouse_ID", how="outer")
# %%
root_path = r"C:\Users\Paula Woest\OneDrive\Desktop\Bachelore Arbeit\Paula"
folder_path = os.path.join(
    root_path, "output_data", "1min_data_feature_matrix_session1.csv"
)

# Save in_roi_ang_dir dataframe as .csv file
combined_features_6.to_csv(folder_path, index=False)
# %%
