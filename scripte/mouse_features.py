# %%
import pandas as pd
import numpy as np

# Parameters for data generation
num_mice = 20  # Number of rows (mice)
num_features = 5  # Number of columns (features)

# Generate unique IDs for each mouse
mouse_ids = [f"Mouse_{i+1}" for i in range(num_mice)]

# Randomly generate feature values
# Example: Generate random float values between 0 and 1
data = np.random.rand(num_mice, num_features)

# Create a DataFrame
feature_columns = [f"Feature_{i+1}" for i in range(num_features)]
df = pd.DataFrame(data, columns=feature_columns)
df.insert(0, "Mouse_ID", mouse_ids)  # Add mouse IDs as the first column

# Save the DataFrame to a CSV file
output_file = "mouse_features.csv"
df.to_csv(output_file, index=False)

# %%
file_path = r"C:\Users\Cystein\Paula\distance_interaction.csv"

df_dist = pd.read_csv(file_path, sep=";")
# Calculate the distance between the Ear and Nose for each frame

# df_dist = pd.DataFrame()
# df_dist['frame_idx'] = df_bl6['frame_idx'].copy()
df_dist[f"{mouse_id}"] = np.sqrt(
    (df_bl6["Nose.x"] - df_cd1["Nose.x"]) ** 2
    + (df_bl6["Nose.y"] - df_cd1["Nose.y"]) ** 2
)
# %%
df_dist.to_csv(file_path, index=False, sep=";")
