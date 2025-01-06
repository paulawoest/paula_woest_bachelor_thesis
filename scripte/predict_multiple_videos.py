# %% Imports
import os
import sleap
from tqdm import tqdm

# %%
input_dir = r"C:\Users\Cystein\Paula\interaction"  # path to directory with video files
models_dir = r"C:\Users\Cystein\Maja\SLEAP\sleap_gui\models"  # path to models directory

# %%
video_files = [files for files in os.listdir(input_dir) if files.endswith(".mp4")]

# %% for bottom-up model
for video_file in video_files:
    # Construct the command
    command = (
        f"sleap-track {os.path.join(input_dir, video_file)} "
        "--max_instances 2 "
        "--tracking.tracker flowmaxtracks "
        "--tracking.max_tracking true "
        "--tracking.max_tracks 2 "
        "--tracking.target_instance_count 2 "
        "--tracking.post_connect_single_breaks 1 "
        f"-o model_v007_{video_file[:-4]}.predictions.slp "
        f"-m {os.path.join(models_dir, '240214_multi_instance_resume_v007.multi_instance')}"
    )

    # Execute the command
    os.system(command)

# %% for top-down id model
for video_file in tqdm(video_files):
    # Construct the command
    command = (
        f"sleap-track {os.path.join(input_dir, video_file)} "
        "--max_instances 2 "  # 1 for single animal
        "--tracking.tracker simplemaxtracks "
        "--tracking.max_tracking true "
        "--tracking.max_tracks 2 "  # 1 for single animal
        "--tracking.target_instance_count 2 "  # 1 for single animal
        "--tracking.post_connect_single_breaks 1 "
        "--tracking.match hungarian "
        "--tracking.track_window 5 "
        "--tracking.similarity iou "
        "--no-empty-frames "
        # change output name
        f"-o model_td_{video_file[:-4]}.predictions.slp "  # [-4] gets rid of .mp4
        # change model names
        f"-m {os.path.join(models_dir, '240404_141957.centroid.n=145')} "
        f"-m {os.path.join(models_dir, '240404_142457.multi_class_topdown.n=145')} "
    )

    # Execute the command
    os.system(command)

# %% for single animal
for video_file in tqdm(video_files):
    # Construct the command
    command = (
        f"sleap-track {os.path.join(input_dir, video_file)} "
        "--max_instances 1 "
        "--tracking.tracker simplemaxtracks "
        "--tracking.max_tracking true "
        "--tracking.max_tracks 1 "
        "--tracking.target_instance_count 1 "
        "--tracking.post_connect_single_breaks 1 "
        f"-o model_td_{video_file[:-4]}.predictions.slp "
        f"-m {os.path.join(models_dir, '240404_141957.centroid.n=145')} "
        f"-m {os.path.join(models_dir, '240404_142457.multi_class_topdown.n=145')} "
    )

    # Execute the command
    os.system(command)

# %%
