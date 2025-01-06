# import the necessary modules
import os
import subprocess


# Get a list of all .mp4 files in the current directory
mp4_files = [f for f in os.listdir() if f.endswith(".mp4")]

# Iterate over each .mp4 file and convert it to grayscale
for file in mp4_files:
    output_file = f"grayscale_{file}"
    subprocess.run(
        [
            "ffmpeg",
            "-i",
            file,
            "-vf",
            "format=gray",
            "-c:v",
            "libx264",
            "-crf",
            "23",
            "-c:a",
            "copy",
            output_file,
        ]
    )
