# %%
import pandas as pd
import os


def create_feature_matrix(csv_folder, output_file, feature_columns, id_column):
    """
    Create a feature matrix from multiple CSV files.

    Parameters:
        csv_folder (str): Path to the folder containing CSV files.
        output_file (str): Path to save the feature matrix CSV.
        feature_columns (list): List of columns to extract as features.
        id_column (str): The column to use as the Mouse_ID.

    Returns:
        pd.DataFrame: The created feature matrix.
    """
    # Initialize an empty list to store data for each mouse
    rows = []

    # Iterate through all files in the folder
    for file in os.listdir(csv_folder):
        if file.endswith(".csv"):  # Process only CSV files
            file_path = os.path.join(csv_folder, file)
            # Read the CSV file
            data = pd.read_csv(file_path)

            # Extract the Mouse_ID and features
            mouse_id = data[id_column].iloc[0]  # Assume Mouse_ID is in the first row
            features = data[feature_columns].iloc[0].values  # Extract feature values

            # Add the Mouse_ID and features as a new row
            rows.append([mouse_id] + list(features))

    # Create the feature matrix DataFrame
    feature_matrix = pd.DataFrame(rows, columns=["mouse_ID"] + feature_columns)

    # Save the feature matrix to a CSV file
    feature_matrix.to_csv(output_file, index=False)

    return feature_matrix


# Example usage:
csv_folder = r"C:\Users\Paula Woest\OneDrive\Desktop\Bachelore Arbeit\Paula\output_data"  # Replace with the path to your folder containing CSV files

output_file = "average_location_session1.csv"

# Define the columns to extract as features and the ID column
feature_columns = [
    "mean_fh_Neck.x",
    "mean_fh_Neck.y",
    "mean_sh_Neck.x",
    "mean_sh_Neck.y",
]  # Adjust based on your CSV files
id_column = "mouse_ID"

# Create the feature matrix
feature_matrix = create_feature_matrix(
    csv_folder, output_file, feature_columns, id_column
)

# Display the resulting feature matrix
print(feature_matrix)

# %%
