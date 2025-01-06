# %% import the necessary modules
import moviepy.editor as moviepy
import os

# %% convert mpg to mp4
FROM_EXT = "mpg"  # file format FROM which we want to convert
TO_EXT = "mp4"  # file format TO which we want to convert
SOURCE_DIR = r"C:\Users\Cystein\Paula\videos_mpg\Cohorte6"  # directory where .mpg files are stored, copy your path here
DEST_DIR = r"C:\Users\Cystein\Paula\videos_mp4\coh6"  # directory where .mp4 files should be stored, copy your path here

for file in os.listdir(SOURCE_DIR):
    if file.lower().endswith(FROM_EXT.lower()):
        from_path = os.path.join(SOURCE_DIR, file)
        to_path = os.path.join(DEST_DIR, file.rsplit(".", 1)[0]) + "." + TO_EXT

        print(f"Converting {from_path} to {to_path}")

        clip = moviepy.VideoFileClip(from_path)
        clip.write_videofile(to_path)

# %%%
