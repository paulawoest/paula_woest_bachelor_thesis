# %% Imports
import os
import shutil


# %% Function to create folders based on ID and session number -- videos
def organize_video_files(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".mp4"):  # Assuming video files are in .mp4 format
            file_parts = filename.split("_")  # Split filename by underscore
            if len(file_parts) >= 4:
                # Extract ID and session number from filename
                ID = file_parts[-2]  # Assuming ID is the second to last part
                session = file_parts[-1].split(".")[0]  # Removing the file extension

                # Create ID folder if it doesn't exist
                id_folder = os.path.join(directory, ID)
                if not os.path.exists(id_folder):
                    os.makedirs(id_folder)

                # Create session subfolder based on session number
                session_folder = os.path.join(id_folder, f"{session}")
                if not os.path.exists(session_folder):
                    os.makedirs(session_folder)

                # Move the file to the session subfolder
                source_file = os.path.join(directory, filename)
                destination_file = os.path.join(session_folder, filename)
                shutil.move(source_file, destination_file)
                print(f"Moved {filename} to {destination_file}")


# %% Function to create folders based on ID and session number -- predictions
def organize_model_files(directory):
    for filename in os.listdir(directory):
        if filename.endswith(
            ".predictions.slp"
        ):  # Assuming video files are in .mp4 format
            file_parts = filename.split("_")  # Split filename by underscore
            if len(file_parts) >= 3:
                # Extract ID and session number from filename
                ID = file_parts[-2]  # Assuming ID is the second to last part
                session = file_parts[-1].split(".")[0]  # Removing the file extension

                # Create ID folder if it doesn't exist
                id_folder = os.path.join(directory, ID)
                if not os.path.exists(id_folder):
                    os.makedirs(id_folder)

                # Create session subfolder based on session number
                session_folder = os.path.join(id_folder, f"{session}")
                if not os.path.exists(session_folder):
                    os.makedirs(session_folder)

                # Move the file to the session subfolder
                source_file = os.path.join(directory, filename)
                destination_file = os.path.join(session_folder, filename)
                shutil.move(source_file, destination_file)
                print(f"Moved {filename} to {destination_file}")


# %%
def organize_csv_files(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):  # Assuming video files are in .mp4 format
            file_parts = filename.split("_")  # Split filename by underscore
            if len(file_parts) >= 5:
                # Extract ID and session number from filename
                ID = file_parts[-2]  # Assuming ID is the second to last part
                session = file_parts[-1].split(".")[0]  # Removing the file extension

                # Create ID folder if it doesn't exist
                id_folder = os.path.join(directory, ID)
                if not os.path.exists(id_folder):
                    os.makedirs(id_folder)

                # Create session subfolder based on session number
                session_folder = os.path.join(id_folder, f"{session}")
                if not os.path.exists(session_folder):
                    os.makedirs(session_folder)

                # Move the file to the session subfolder
                source_file = os.path.join(directory, filename)
                destination_file = os.path.join(session_folder, filename)
                shutil.move(source_file, destination_file)
                print(f"Moved {filename} to {destination_file}")


# %%
directory = r"C:\Users\Cystein\Paula"  # adjust directory accordingly, i.e. your SLEAP_Project folder
organize_video_files(directory)
organize_model_files(directory)
organize_csv_files(directory)
# %%
