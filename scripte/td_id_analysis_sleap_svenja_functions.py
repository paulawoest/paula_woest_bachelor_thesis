# %% # importing libraries
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import scipy.interpolate
import tkinter as tk
from datetime import datetime
from tkinter import messagebox
from PIL import Image, ImageTk

# TODO: add rest of functions from all other files
# TODO: add description of what each function does with """xyz"""


# %%
def get_experiment_info():
    """
    Pop-up window for user-specified experimental information

    Returns:
        experimenter_name
        mouse_id
        session
        experiment_id
        analyst_name
        analysis_date
    """
    experiment_info = {}

    def store_input():
        experimenter_name = entry_widgets[0].get()
        mouse_id = entry_widgets[1].get()
        session = entry_widgets[2].get()
        experiment_id = entry_widgets[3].get()
        analyst_name = entry_widgets[4].get()
        cohort = entry_widgets[5].get()
        analysis_date = datetime.now().strftime("%Y-%m-%d")

        # Check if experimenter name and analyst name consist of two initials
        if not (is_initials(experimenter_name) and is_initials(analyst_name)):
            messagebox.showerror(
                "Error", "Name of Experimenter and Analyst must be two initials."
            )
            return
        # Check if mouse ID consists of exactly four digits
        elif not is_valid_mouse_id(mouse_id):
            messagebox.showerror("Error", "Mouse ID must be exactly four digits.")
            return

        # Store the input in the experiment_info dictionary
        experiment_info.update(
            {
                "Experimenter Name": experimenter_name,
                "Mouse ID": mouse_id,
                "Session": session,
                "Experiment ID": experiment_id,
                "Analyst Name": analyst_name,
                "Cohort": cohort,
                "Analysis Date": analysis_date,
            }
        )

        root.destroy()

    def is_initials(name):
        return len(name) == 2 and name.isalpha()

    def is_valid_mouse_id(mouse_id):
        return mouse_id.isdigit() and len(mouse_id) == 4

    # Create the main window
    root = tk.Tk()
    root.title("Enter Experiment Information")

    # Set the size of the window
    root.geometry("300x300")

    # Create labels and entry widgets for each input
    labels = [
        "Who conducted the Experiment?",
        "Mouse ID:",
        "Session",
        "Experiment ID:",
        "Who is Analysing the Data?",
        "Cohort:",
    ]
    entry_widgets = []

    for label_text in labels:
        label = tk.Label(root, text=label_text)
        label.pack()
        entry_widget = tk.Entry(root)
        entry_widget.pack()
        entry_widgets.append(entry_widget)

    # Create a button to store the input
    button = tk.Button(root, text="Store Input", command=store_input)
    button.pack()

    # Run the tkinter event loop
    root.mainloop()

    # Return the experiment_info dictionary
    return experiment_info if experiment_info else None


# %% read data and split in tracks
def split_data(csv_path):
    df = pd.read_csv(csv_path)

    # split dataframes according to tracks
    df_bl6 = df[df["track"] == "bl6"].copy()
    df_cd1 = df[df["track"] == "cd1"].copy()

    return df_bl6, df_cd1


# %%
def add_missing_frames(df, possible_frames=9107):
    """
    Fill Missing Frames

    Args:
        df: dataframe with incomplete or missing frame numbers
        possible_frames: number of possible frames, i.e., number of frames in respective video

    Returns:
        df: dataframe with as many frames as respective video
    """
    # Get unique frame numbers present in the DataFrame
    frame_numbers_present = df["frame_idx"].unique()

    # Create a reference list of all possible frame numbers
    all_frame_numbers = np.arange(possible_frames)

    # Find the missing frame numbers
    missing_frame_numbers = np.setdiff1d(all_frame_numbers, frame_numbers_present)

    # Create a DataFrame with the missing frame numbers
    missing_frames_df = pd.DataFrame({"frame_idx": missing_frame_numbers})

    # Concatenate the missing frames DataFrame with the original DataFrame
    df = pd.concat([df, missing_frames_df], ignore_index=True)

    # Sort the DataFrame by frame_idx
    df = df.sort_values("frame_idx")

    # Reset the index
    df = df.reset_index(drop=True)

    return df


# %%
def scale_column(df, column_name, new_min=0, new_max=1):
    """
    Scales column to new possible values

    Args:
        df: dataframe
        column_name: name of the column that should be scaled
        new_min: minimum value of what you want column to scale to
        new_max: maximum value of what you want column to scale to

    Returns:
        df: dataframe with additional scaled column
    """
    min_value = 0  # minimum possible value of original data
    max_value = 1.5  # maximum possible value of original data

    df[f"{column_name}_scaled"] = (df[column_name] - min_value) / (
        max_value - min_value
    ) * (new_max - new_min) + new_min

    return df


# %%
# TODO: add colour as an argument, if df.name == 'bl6' blue, else orange
def plot_histogram_with_percentile(
    data,
    column_name,
    folder_path,
    file_name_suffix,
    experimenter_name,
    mouse_id,
    session,
    experiment_id,
    analyst_name,
    analysis_date,
    percentile=0.20,
    num_bins=20,
):
    """
    Plot a Histogram with Dotted Percentile Line

    Args:
        data: dataframe you want to plot
        column_name: column you want to plot
        folder_path: folder path to where the .svg should be saved
        file_name_suffix: file name suffix
        experimenter_name
        mouse_id
        session
        experiment_id
        analyst_name
        analysis_date
        percentile: dotted percentile line, default 0.20
        num_bins: number of bins, default 20

    Returns:
        plot of histogram with percentile + File saved as .svg in specified folder path
    """
    # Calculate the specified percentile
    percentile_value = data[column_name].quantile(percentile)

    # Plot the histogram
    plt.hist(data[column_name], bins=num_bins)

    # Add a vertical line at the specified percentile
    plt.axvline(
        x=percentile_value,
        color="red",
        linestyle="--",
        label=f"{percentile*100:.0f}th percentile",
    )

    # Add labels and title
    plt.xlabel(column_name)
    plt.ylabel("Frequency")
    plt.title(f"Histogram of {column_name} of {data.name}")

    # Add a legend
    plt.legend()

    file_name = f"{experimenter_name}_{mouse_id}_{session}_{experiment_id}_{analyst_name}_{analysis_date}_histogram_{file_name_suffix}.svg"
    save_path = os.path.join(folder_path, file_name)

    # Save the plot
    plt.savefig(save_path)

    # Show the plot
    plt.show()


# %%
def clean_data(threshold_bl6, threshold_cd1, df_bl6, df_cd1):
    # threshold to sort out data with low certainty
    columns_to_clean = df_bl6.columns.difference(["track", "frame_idx"])
    df_bl6.loc[df_bl6["Neck.score"] <= threshold_bl6, columns_to_clean] = np.nan
    df_cd1.loc[df_cd1["Neck.score"] <= threshold_cd1, columns_to_clean] = np.nan

    return df_bl6, df_cd1


# %%
# TODO: add colour as an argument, if df.name == 'bl6' blue, else orange
# TODO: make as one histogram function, percentile = None as default argument, if specified, add axvline and legend
def plot_histogram_without_percentile(
    data,
    column_name,
    folder_path,
    file_name_suffix,
    experimenter_name,
    mouse_id,
    session,
    experiment_id,
    analyst_name,
    analysis_date,
    num_bins=20,
):

    # Plot the histogram
    plt.hist(data[column_name], bins=num_bins)

    # Add labels and title
    plt.xlabel(column_name)
    plt.ylabel("Frequency")
    plt.title(f"Histogram of {column_name} of {data.name}")

    file_name = f"{experimenter_name}_{mouse_id}_{session}_{experiment_id}_{analyst_name}_{analysis_date}_histogram_{file_name_suffix}.svg"
    save_path = os.path.join(folder_path, file_name)

    # Save the plot
    plt.savefig(save_path)
    # Show the plot
    plt.show()


# %%
def interpolate_track_frame(df_bl6, df_cd1):
    # set track again
    df_bl6["track"] = "bl6"
    df_cd1["track"] = "cd1"

    # linear interpolation of frames
    df_bl6["frame_idx"] = df_bl6["frame_idx"].interpolate(method="linear")
    df_cd1["frame_idx"] = df_cd1["frame_idx"].interpolate(method="linear")

    return df_bl6, df_cd1


# %%
def moving_median_filter(df, column, window=10):
    """
    Moving Median Filter

    Args:
        df: dataframe
        column: column(s) over which moving median filter should be applied
        window: window size of kernel, default 10

    Returns:
        df: dataframe
    """
    df[column] = df[column].rolling(window=window, min_periods=1, center=True).median()

    return df


# %% interpolation nearest
def interpolate_column(column):
    """
    Linear Interpolation of specified column

    Args:
        column: column(s) with nan values that need to be interpolated

    Returns:
        interpolated_series: Series object of interpolated values
    """
    non_nan_indices = column.index[column.notna()]
    non_nan_values = column.loc[non_nan_indices]

    x_new = column.index[
        (column.index >= non_nan_indices.min())
        & (column.index <= non_nan_indices.max())
    ]
    f = scipy.interpolate.interp1d(
        non_nan_indices, non_nan_values, kind="linear"
    )  # instead of nearest

    # Generate interpolated values for all indices
    interpolated_values = f(x_new)

    # Create a Series with the interpolated values aligned with the original DataFrame's index
    interpolated_series = pd.Series(interpolated_values, index=x_new)

    return interpolated_series


# %%
def interpolate_missing_values(df, columns_to_interpolate):
    """
    Apply interpolate_column function to dataframe

    Args:
        df: dataframe
        columns_to_interpolate: column(s) with nan values that need to be interpolated

    Returns:
        df: dataframe with interpolated values
    """
    # Apply interpolation only to specified columns
    df[columns_to_interpolate] = df[columns_to_interpolate].apply(interpolate_column)

    return df


# %% normal map of where mouse was, no information about time
def plot_spatial(
    df1,
    df2,
    x_column,
    y_column,
    title,
    folder_path,
    experimenter_name,
    mouse_id,
    session,
    experiment_id,
    analyst_name,
    analysis_date,
):
    plt.plot(df1[x_column], df1[y_column], color="blue", label=df1.name)

    # Plot df_cd1 data with blue color
    plt.plot(df2[x_column], df2[y_column], color="orange", label=df2.name)

    # Set the limits of the x-axis and y-axis
    plt.xlim(0, 1280)
    plt.ylim(1024, 0)  # Inverted y-axis to match video

    # Add labels and title
    plt.xlabel(f"{x_column} coordinate")
    plt.ylabel(f"{y_column} coordinate")
    plt.title(f"Spatial Plot of {title}")

    # Add legend
    plt.legend()

    # Save the cropped frame
    file_name = f"{experimenter_name}_{mouse_id}_{session}_{experiment_id}_{analyst_name}_{analysis_date}_spatial_plot_of_{title}.svg"
    save_path = os.path.join(folder_path, file_name)

    # Save the plot
    plt.savefig(save_path)

    # Show the plot
    plt.show()


# %%
def plot_spatial_sess1(
    df1,
    x_column,
    y_column,
    title,
    folder_path,
    experimenter_name,
    mouse_id,
    session,
    experiment_id,
    analyst_name,
    analysis_date,
):
    """
    Spatial Plot of specified Nodes/Body Parts

    Args:
        df1: dataframe
        x_column: x-coordinate of node/body part
        y_column: y-coordinate of node/body part
        title: title of plot
        folder_path
        experimenter_name
        mouse_id
        session
        experiment_id
        analyst_name
        analysis_date
    """
    plt.plot(df1[x_column], df1[y_column], color="blue", label=df1.name)

    # Set the limits of the x-axis and y-axis
    plt.xlim(0, 1280)
    plt.ylim(1024, 0)  # Inverted y-axis to match video

    # Add labels and title
    plt.xlabel(f"{x_column} coordinate")
    plt.ylabel(f"{y_column} coordinate")
    plt.title(f"Spatial Plot of {title}")

    # Add legend
    plt.legend()

    # Save the cropped frame
    file_name = f"{experimenter_name}_{mouse_id}_{session}_{experiment_id}_{analyst_name}_{analysis_date}_spatial_plot_of_{title}.svg"
    save_path = os.path.join(folder_path, file_name)

    # Save the plot
    plt.savefig(save_path)

    # Show the plot
    plt.show()


# %% colourmap of when mouse was where
def plot_temporal_spatial(
    df,
    x_column,
    y_column,
    frame_column,
    title,
    folder_path,
    experimenter_name,
    mouse_id,
    session,
    experiment_id,
    analyst_name,
    analysis_date,
):
    """
    Temporal Spatial Plot of specified Nodes/Body Parts

    Args:
        df: dataframe
        x_column: x-coordinate of node/body part
        y_column: y-coordinate of node/body part
        frame_column: frame_idx column for colourmap
        title: title of plot
        folder_path
        experimenter_name
        mouse_id
        session
        experiment_id
        analyst_name
        analysis_date
    """
    cmap = plt.cm.Spectral  # You can choose any other colormap

    # Plot the data with colormap based on frame index
    plt.scatter(
        df[x_column], df[y_column], c=df[frame_column], cmap=cmap, s=5
    )  # s is size of points

    # Set the colorbar to indicate frame index values
    cbar = plt.colorbar()
    cbar.set_label("Frame index")

    # Set the limits of the x-axis and y-axis
    plt.xlim(0, 1280)
    plt.ylim(1024, 0)  # Inverted y-axis to match video

    # Add labels and title
    plt.xlabel(f"{x_column} coordinate")
    plt.ylabel(f"{y_column} coordinate")
    plt.title(f"Temporal Spatial Plot of {title}")

    file_name = f"{experimenter_name}_{mouse_id}_{session}_{experiment_id}_{analyst_name}_{analysis_date}_temporal_spatial_plot_of_{title}.svg"
    save_path = os.path.join(folder_path, file_name)

    # Save the plot
    plt.savefig(save_path)


# %%
def display_instructions(image_path):
    """
    pop-up window to display instructions on where to click for ROI
    """

    def close_window(event=None):
        root.destroy()

    # Create the main window
    root = tk.Tk()

    # Set the window title
    root.title("Instructions on Where to Click for ROI")

    # Get the screen width and height
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()

    # Set the window size
    window_width = int(screen_width * 0.55)  # 55% of the screen width
    window_height = int(screen_height * 0.9)  # 90% of the screen height

    # Calculate the x and y coordinates for the window to be centered
    x = (screen_width - window_width) // 2
    y = (screen_height - window_height) // 2

    # Set the window size and position
    root.geometry(f"{window_width}x{window_height}+{x}+{y}")

    # Create a label widget to display the text
    text = "The image below is an EXAMPLE of where you need to click to specify the region of interest.\n\nMemorise the EXACT LOCATION and ORDER of the clicks as shown below.\n\nANOTHER WINDOW will appear, there you need to left click\nthe FOUR MARKED corners in the ORDER specified in the example below.\n\nOnce memorised, close this window."
    label_text = tk.Label(root, text=text, padx=20, pady=20, font=("Arial", 12))
    label_text.pack()

    # Load the image
    image = Image.open(image_path)
    image.thumbnail((window_width, window_height - 250))  # Thumbnail to fit the window

    # Convert the image to a format that tkinter can use
    tk_image = ImageTk.PhotoImage(image)

    # Create a label widget to display the image
    label_image = tk.Label(root, image=tk_image)
    label_image.pack()

    # Bind the close_window function to the window close button click event
    root.protocol("WM_DELETE_WINDOW", close_window)
    root.lift()  # Bring the window to the front

    # Run the tkinter event loop
    root.mainloop()


# %%
def click_event(event, x, y, flags, params):
    """
    Function to record x- and y-coordinates of left mouse-click
    """
    if event == cv2.EVENT_LBUTTONDOWN:
        # Draw a red circle at the clicked point
        cv2.circle(params["frame"], (x, y), 4, (0, 0, 255), -1)
        cv2.imshow("Frame", params["frame"])
        params["left"].append((x, y))
        print(x, y)

        # Check if the desired number of clicks is reached
        if len(params["left"]) >= 4:
            # Wait for 2 milliseconds
            cv2.waitKey(500)
            cv2.destroyAllWindows()

    if event == cv2.EVENT_RBUTTONDOWN:
        params["right"].append((x, y))
        print(x, y)


# %%
def display_frame(video_path, frame_number):
    """
    display random frame from video where users click cornes of box to determine ROI
    """
    # open video file#
    vid = cv2.VideoCapture(video_path)

    # set frame position
    vid.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    # read frame
    success, frame = vid.read()
    # greyFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # check frame read successful
    if success:
        # display frame
        # cv2.imshow("Frame", greyFrame)
        cv2.imshow("Frame", frame)

        # Create a dictionary to store coordinates and the frame
        # coordinates = {"frame": greyFrame, "left": [], "right": []}
        coordinates = {"frame": frame, "left": [], "right": []}

        cv2.setMouseCallback("Frame", click_event, coordinates)

        # Loop to wait for key press or window close event
        while True:
            key = cv2.waitKey(1)  # Wait for 1 millisecond
            if key & 0xFF == ord("q"):  # Check if 'q' key is pressed
                break  # Exit the loop if 'q' key is pressed
            if (
                cv2.getWindowProperty("Frame", cv2.WND_PROP_VISIBLE) < 1
            ):  # Check if window is closed
                break  # Exit the loop if window is closed
    else:
        print(f"Error reading frame {frame_number}")

    # release video capture object
    vid.release()

    # close all OpenCV windows
    cv2.destroyAllWindows()

    return coordinates


# %%
def calculate_rectangle(left):
    """
    calculate rectangle based on user-defined coordinates of box/ROI

    Returns:
        x- and y-coordinates of left top and bottom right corner of rectangle
    """

    # find middle point between bottom corners
    midB = np.zeros((2), dtype=np.float64)
    midB[0] = (left[0, 0] + left[2, 0]) / 2  # x coordinates
    midB[1] = (left[0, 1] + left[2, 1]) / 2  # y coordinates

    # find middle point between top corners
    midT = np.zeros((2), dtype=np.float64)
    midT[0] = (left[1, 0] + left[3, 0]) / 2
    midT[1] = (left[1, 1] + left[3, 1]) / 2

    # y coordinate midT-midB
    y_mid_len = midT[1] - midB[1]

    # x coordinate rightB-leftB
    x_bot_len = left[2, 0] - left[0, 0]

    # corner points rectangle
    xleftT = midB[0] - x_bot_len
    xrightB = midB[0] + x_bot_len
    yrightB = 0
    yleftT = midB[1] + y_mid_len

    return xleftT, yleftT, xrightB, yrightB


# %%
def display_rectangle(video_path, frame_number, xleftT, yleftT, xrightB, yrightB):
    """
    display random frame with drawn rectangle to make sure that ROI was created successfully
    """
    # open video file#
    vid = cv2.VideoCapture(video_path)

    # set frame position
    vid.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    # read frame
    success, frame = vid.read()
    greyFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # check frame read successful
    if success:
        cv2.rectangle(
            greyFrame,
            (int(xleftT), int(yleftT)),
            (int(xrightB), int(yrightB)),
            (0, 255, 0),
            2,
        )
        # display frame
        cv2.imshow("Frame", greyFrame)

        # Loop to wait for key press or window close event
        while True:
            key = cv2.waitKey(1)  # Wait for 1 millisecond
            if key & 0xFF == ord("q"):  # Check if 'q' key is pressed
                break  # Exit the loop if 'q' key is pressed
            if (
                cv2.getWindowProperty("Frame", cv2.WND_PROP_VISIBLE) < 1
            ):  # Check if window is closed
                break  # Exit the loop if window is closed
    else:
        print(f"Error reading frame {frame_number}")

    # release video capture object
    vid.release()

    # close all OpenCV windows
    cv2.destroyAllWindows()


# %% empty frame matrix
def empty_frame_tensor(video_path):
    vid = cv2.VideoCapture(video_path)
    mask_frames = np.zeros(
        (int((vid.get(cv2.CAP_PROP_FRAME_COUNT))), 3, 4), np.float64
    )  # 3 cause 3 skeleton nodes, 4 cause 2 dimensions x 2 boolean (True/False)
    return mask_frames


# %%
def rectangle_and_save(
    video_path,
    frame_number,
    xleftT,
    yleftT,
    xrightB,
    yrightB,
    folder_path,
    experimenter_name,
    mouse_id,
    session,
    experiment_id,
    analyst_name,
    analysis_date,
):
    """
    Create .jpg files of frame + ROI rectangle
    """
    # open video file#
    vid = cv2.VideoCapture(video_path)

    # set frame position
    vid.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    # read frame
    success, frame = vid.read()
    greyFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # check frame read successful
    if success:
        cv2.rectangle(
            greyFrame,
            (int(xleftT), int(yleftT)),
            (int(xrightB), int(yrightB)),
            (0, 255, 0),
            2,
        )
        # display frame
        # cv2.imshow('Frame', greyFrame)

        # Create file name
        file_name = f"{experimenter_name}_{mouse_id}_{session}_{experiment_id}_{analyst_name}_{analysis_date}_frame{frame_number}.jpg"

        # Specify the file path for saving
        output_path = os.path.join(folder_path, file_name)

        # Save the frame to the specified output path
        cv2.imwrite(output_path, greyFrame)
        # cv2.waitKey(0) # wait indefinitely for key press
    else:
        print(f"Error reading frame {frame_number}")

    # release video capture object
    vid.release()

    # close all OpenCV windows
    cv2.destroyAllWindows()


# %%
def plot_frames_control(
    saved_frames,
    random_frames,
    folder_path,
    file_name_suffix,
    experimenter_name,
    mouse_id,
    session,
    experiment_id,
    analyst_name,
    analysis_date,
    num_rows=3,
    num_cols=3,
):
    """
    plot 3x3 saved frames of bl6 mouse in roi
    """
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(12, 12))

    # Plot each frame in a subplot
    for i, frame in enumerate(saved_frames):
        row = i // num_cols
        col = i % num_cols
        axs[row, col].imshow(
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        )  # OpenCV reads in BGR format, convert to RGB for display
        axs[row, col].set_title(f"Frame Index: {random_frames[i]}")
        axs[row, col].axis("off")

    plt.tight_layout()

    file_name = f"{experimenter_name}_{mouse_id}_{session}_{experiment_id}_{analyst_name}_{analysis_date}_{num_rows}_x_{num_cols}_control_plot_of_{file_name_suffix}.svg"
    save_path = os.path.join(folder_path, file_name)

    # Save the plot
    plt.savefig(save_path)

    plt.show()


# %% Function to calculate the angle between two vectors
def angle_between_vectors(vector1, vector2):
    """
    calculates angle between two vectors
    """
    dot_product = np.sum(vector1 * vector2, axis=1)
    magnitude_product = np.linalg.norm(vector1, axis=1) * np.linalg.norm(
        vector2, axis=1
    )
    cosine_angle = dot_product / magnitude_product
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)


# %%
def calculate_cross_product(vector1, vector2):
    """
    calculates cross products between two vectors
    """
    return np.cross(vector1, vector2)


# %%
def draw_frame_with_vector(
    video_path, df_bl6, df_cd1, frameN, xleftT, yleftT, xrightB, yrightB
):
    """
    display frame with origin vectors and neck nose, nose neck vectors
    """
    # Load the DataFrame with coordinates
    x_neck_bl6 = df_bl6["Neck.x"][frameN]
    y_neck_bl6 = df_bl6["Neck.y"][frameN]
    x_nose_bl6 = df_bl6["Nose.x"][frameN]
    y_nose_bl6 = df_bl6["Nose.y"][frameN]

    x_neck_cd1 = df_cd1["Neck.x"][frameN]
    y_neck_cd1 = df_cd1["Neck.y"][frameN]

    # Origin vectors
    x_origin = 0
    y_origin = 0

    # Load video file
    vid = cv2.VideoCapture(video_path)

    vid.set(cv2.CAP_PROP_POS_FRAMES, frameN)

    # Read the first frame
    success, frame = vid.read()
    if not success:
        raise ValueError("Error reading video file")

    # Draw the vectors on the frame
    cv2.arrowedLine(
        frame,
        (int(x_neck_bl6), int(y_neck_bl6)),
        (int(x_nose_bl6), int(y_nose_bl6)),
        (0, 0, 255),
        2,
        tipLength=0.5,
    )  # Red line from neck to nose
    cv2.circle(
        frame, (int(x_neck_bl6), int(y_neck_bl6)), 1, (255, 0, 0), -1
    )  # Blue dot for neck
    cv2.circle(
        frame, (int(x_nose_bl6), int(y_nose_bl6)), 1, (0, 255, 0), -1
    )  # Green dot for nose

    cv2.arrowedLine(
        frame,
        (int(x_nose_bl6), int(y_nose_bl6)),
        (int(x_neck_cd1), int(y_neck_cd1)),
        (0, 0, 255),
        2,
        tipLength=0.1,
    )  # Red line from neck to nose
    cv2.circle(
        frame, (int(x_nose_bl6), int(y_nose_bl6)), 1, (200, 0, 0), -1
    )  # Blue dot for neck
    cv2.circle(
        frame, (int(x_neck_cd1), int(y_neck_cd1)), 1, (0, 200, 0), -1
    )  # Green dot for nose

    # Origin vectors
    cv2.arrowedLine(
        frame,
        (int(x_origin), int(y_origin)),
        (int(x_neck_cd1), int(y_neck_cd1)),
        (0, 255, 0),
        2,
        tipLength=0.01,
    )  # Red line from neck to nose
    cv2.circle(
        frame, (int(x_origin), int(y_origin)), 3, (255, 0, 0), -1
    )  # Blue dot for neck
    cv2.circle(
        frame, (int(x_neck_cd1), int(y_neck_cd1)), 3, (255, 0, 0), -1
    )  # Green dot for nose

    cv2.arrowedLine(
        frame,
        (int(x_origin), int(y_origin)),
        (int(x_nose_bl6), int(y_nose_bl6)),
        (0, 255, 0),
        2,
        tipLength=0.01,
    )  # Red line from neck to nose
    cv2.circle(
        frame, (int(x_origin), int(y_origin)), 3, (255, 0, 0), -1
    )  # Blue dot for neck
    cv2.circle(frame, (int(x_nose_bl6), int(y_nose_bl6)), 3, (255, 0, 0), -1)

    cv2.arrowedLine(
        frame,
        (int(x_origin), int(y_origin)),
        (int(x_neck_bl6), int(y_neck_bl6)),
        (0, 255, 0),
        2,
        tipLength=0.01,
    )  # Red line from neck to nose
    cv2.circle(
        frame, (int(x_origin), int(y_origin)), 3, (255, 0, 0), -1
    )  # Blue dot for neck
    cv2.circle(frame, (int(x_neck_bl6), int(y_neck_bl6)), 3, (255, 0, 0), -1)

    cv2.rectangle(
        frame, (int(xleftT), int(yleftT)), (int(xrightB), int(yrightB)), (0, 0, 0), 2
    )

    # Display the frame with the vector
    cv2.imshow("Frame with Vector", frame)
    cv2.waitKey(0)  # Wait indefinitely for a key press
    cv2.destroyAllWindows()  # Close all OpenCV windows


# %%
def draw_frame_with_vector_sess1(
    video_path,
    v_x_mid_box,
    v_y_mid_box,
    df_bl6,
    frameN,
    xleftT,
    yleftT,
    xrightB,
    yrightB,
):
    """
    display frame with origin vectors and neck nose, nose mid_box vectors
    """
    # Load the DataFrame with coordinates
    x_neck_bl6 = df_bl6["Neck.x"][frameN]
    y_neck_bl6 = df_bl6["Neck.y"][frameN]
    x_nose_bl6 = df_bl6["Nose.x"][frameN]
    y_nose_bl6 = df_bl6["Nose.y"][frameN]

    # Origin vectors
    x_origin = 0
    y_origin = 0

    # Load video file
    vid = cv2.VideoCapture(video_path)

    vid.set(cv2.CAP_PROP_POS_FRAMES, frameN)

    # Read the first frame
    success, frame = vid.read()
    if not success:
        raise ValueError("Error reading video file")

    # Draw the vectors on the frame
    cv2.arrowedLine(
        frame,
        (int(x_neck_bl6), int(y_neck_bl6)),
        (int(x_nose_bl6), int(y_nose_bl6)),
        (0, 0, 255),
        2,
        tipLength=0.5,
    )  # Red line from neck to nose
    cv2.circle(
        frame, (int(x_neck_bl6), int(y_neck_bl6)), 1, (255, 0, 0), -1
    )  # Blue dot for neck
    cv2.circle(
        frame, (int(x_nose_bl6), int(y_nose_bl6)), 1, (0, 255, 0), -1
    )  # Green dot for nose

    cv2.arrowedLine(
        frame,
        (int(x_nose_bl6), int(y_nose_bl6)),
        (int(v_x_mid_box), int(v_y_mid_box)),
        (0, 0, 255),
        2,
        tipLength=0.1,
    )  # Red line from neck to nose
    cv2.circle(
        frame, (int(x_nose_bl6), int(y_nose_bl6)), 1, (200, 0, 0), -1
    )  # Blue dot for neck
    cv2.circle(
        frame, (int(v_x_mid_box), int(v_y_mid_box)), 1, (0, 200, 0), -1
    )  # Green dot for nose

    # Origin vectors
    cv2.arrowedLine(
        frame,
        (int(x_origin), int(y_origin)),
        (int(v_x_mid_box), int(v_y_mid_box)),
        (0, 255, 0),
        2,
        tipLength=0.01,
    )  # Red line from neck to nose
    cv2.circle(
        frame, (int(x_origin), int(y_origin)), 3, (255, 0, 0), -1
    )  # Blue dot for neck
    cv2.circle(
        frame, (int(v_x_mid_box), int(v_y_mid_box)), 3, (255, 0, 0), -1
    )  # Green dot for nose

    cv2.arrowedLine(
        frame,
        (int(x_origin), int(y_origin)),
        (int(x_nose_bl6), int(y_nose_bl6)),
        (0, 255, 0),
        2,
        tipLength=0.01,
    )  # Red line from neck to nose
    cv2.circle(
        frame, (int(x_origin), int(y_origin)), 3, (255, 0, 0), -1
    )  # Blue dot for neck
    cv2.circle(frame, (int(x_nose_bl6), int(y_nose_bl6)), 3, (255, 0, 0), -1)

    cv2.arrowedLine(
        frame,
        (int(x_origin), int(y_origin)),
        (int(x_neck_bl6), int(y_neck_bl6)),
        (0, 255, 0),
        2,
        tipLength=0.01,
    )  # Red line from neck to nose
    cv2.circle(
        frame, (int(x_origin), int(y_origin)), 3, (255, 0, 0), -1
    )  # Blue dot for neck
    cv2.circle(frame, (int(x_neck_bl6), int(y_neck_bl6)), 3, (255, 0, 0), -1)

    cv2.rectangle(
        frame, (int(xleftT), int(yleftT)), (int(xrightB), int(yrightB)), (0, 0, 0), 2
    )

    # Display the frame with the vector
    cv2.imshow("Frame with Vector", frame)
    cv2.waitKey(0)  # Wait indefinitely for a key press
    cv2.destroyAllWindows()  # Close all OpenCV windows


# %%
def rectangle_ang_dir_and_save(
    video_path,
    frameN,
    xleftT,
    yleftT,
    xrightB,
    yrightB,
    folder_path,
    experimenter_name,
    mouse_id,
    session,
    experiment_id,
    analyst_name,
    analysis_date,
    x_neck_bl6,
    y_neck_bl6,
    x_nose_bl6,
    y_nose_bl6,
    x_neck_cd1,
    y_neck_cd1,
    origin=False,
    crop_margin=100,
):
    """
    Create .jpg files of frame + ROI rectangle + vector neck nose and nose + neck
    """
    x_origin = 0
    y_origin = 0

    # Open video file
    vid = cv2.VideoCapture(video_path)

    # Set frame position
    vid.set(cv2.CAP_PROP_POS_FRAMES, frameN)

    # Read frame
    success, frame = vid.read()

    # Check frame read successful
    if success:
        # Draw the vector on the frame
        cv2.arrowedLine(
            frame,
            (int(x_neck_bl6), int(y_neck_bl6)),
            (int(x_nose_bl6), int(y_nose_bl6)),
            (255, 0, 0),
            2,
            tipLength=0.5,
        )  # Red line from neck to nose
        cv2.circle(
            frame, (int(x_neck_bl6), int(y_neck_bl6)), 2, (255, 0, 0), -1
        )  # Blue dot for neck
        # cv2.circle(frame, (int(x_nose_bl6), int(y_nose_bl6)), 1, (0, 255, 0), -1)  # blue dot for nose

        # bl6 nose to cd1 neck
        cv2.arrowedLine(
            frame,
            (int(x_nose_bl6), int(y_nose_bl6)),
            (int(x_neck_cd1), int(y_neck_cd1)),
            (0, 165, 255),
            2,
            tipLength=0.3,
        )  # Red line from neck to nose
        cv2.circle(
            frame, (int(x_nose_bl6), int(y_nose_bl6)), 2, (255, 0, 0), -1
        )  # Blue dot for nose
        cv2.circle(
            frame, (int(x_neck_cd1), int(y_neck_cd1)), 2, (0, 165, 255), -1
        )  # orange dot for neck

        cv2.rectangle(
            frame,
            (int(xleftT), int(yleftT)),
            (int(xrightB), int(yrightB)),
            (0, 0, 0),
            2,
        )

        # Create mask for the rectangle area
        mask = np.zeros_like(frame)
        cv2.rectangle(
            mask,
            (int(xleftT), int(yleftT)),
            (int(xrightB), int(yrightB)),
            (255, 255, 255),
            -1,
        )  # White rectangle

        # Apply mask to the frame
        masked_frame = cv2.bitwise_and(frame, mask)

        # Find contours in the masked frame
        contours, _ = cv2.findContours(
            masked_frame[:, :, 0], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Find bounding box of non-black pixels with margin
        if contours:
            x, y, w, h = cv2.boundingRect(contours[0])
            x -= crop_margin
            y -= crop_margin
            w += 2 * crop_margin
            h += 2 * crop_margin
            cropped_frame = frame[
                max(0, y) : min(frame.shape[0], y + h),
                max(0, x) : min(frame.shape[1], x + w),
            ]

        else:
            cropped_frame = frame

        # Draw vectors on the cropped frame
        if origin:
            # Draw origin vectors
            cv2.arrowedLine(
                cropped_frame,
                (int(x_origin), int(y_origin)),
                (int(x_neck_cd1), int(y_neck_cd1)),
                (0, 255, 0),
                2,
                tipLength=0.01,
            )  # Red line from neck to nose
            cv2.arrowedLine(
                cropped_frame,
                (int(x_origin), int(y_origin)),
                (int(x_nose_bl6), int(y_nose_bl6)),
                (0, 255, 0),
                2,
                tipLength=0.01,
            )  # Red line from neck to nose
            cv2.arrowedLine(
                cropped_frame,
                (int(x_origin), int(y_origin)),
                (int(x_neck_bl6), int(y_neck_bl6)),
                (0, 255, 0),
                2,
                tipLength=0.01,
            )  # Red line from neck to nose

        # Save the cropped frame
        file_name = f"{experimenter_name}_{mouse_id}_{session}_{experiment_id}_{analyst_name}_{analysis_date}_roi_ang_dir_frame{frameN}.jpg"
        output_path = os.path.join(folder_path, file_name)
        cv2.imwrite(output_path, cropped_frame)
    else:
        print(f"Error reading frame {frameN}")

    # Release video capture object
    vid.release()

    # Close all OpenCV windows
    cv2.destroyAllWindows()

    return None


# %%
def rectangle_ang_dir_and_save_sess1(
    video_path,
    frameN,
    xleftT,
    yleftT,
    xrightB,
    yrightB,
    folder_path,
    experimenter_name,
    mouse_id,
    session,
    experiment_id,
    analyst_name,
    analysis_date,
    x_neck_bl6,
    y_neck_bl6,
    x_nose_bl6,
    y_nose_bl6,
    v_x_mid_box,
    v_y_mid_box,
    origin=False,
    crop_margin=100,
):
    """
    Create .jpg files of frame + ROI rectangle + neck nose and nose mid_box vector
    """
    x_origin = 0
    y_origin = 0

    # Open video file
    vid = cv2.VideoCapture(video_path)

    # Set frame position
    vid.set(cv2.CAP_PROP_POS_FRAMES, frameN)

    # Read frame
    success, frame = vid.read()

    # Check frame read successful
    if success:
        # Draw the vector on the frame
        cv2.arrowedLine(
            frame,
            (int(x_neck_bl6), int(y_neck_bl6)),
            (int(x_nose_bl6), int(y_nose_bl6)),
            (255, 0, 0),
            2,
            tipLength=0.5,
        )  # Red line from neck to nose
        cv2.circle(
            frame, (int(x_neck_bl6), int(y_neck_bl6)), 2, (255, 0, 0), -1
        )  # Blue dot for neck
        # cv2.circle(frame, (int(x_nose_bl6), int(y_nose_bl6)), 1, (0, 255, 0), -1)  # blue dot for nose

        # bl6 nose to cd1 neck
        cv2.arrowedLine(
            frame,
            (int(x_nose_bl6), int(y_nose_bl6)),
            (int(v_x_mid_box), int(v_y_mid_box)),
            (0, 165, 255),
            2,
            tipLength=0.3,
        )  # Red line from neck to nose
        cv2.circle(
            frame, (int(x_nose_bl6), int(y_nose_bl6)), 2, (255, 0, 0), -1
        )  # Blue dot for nose
        cv2.circle(
            frame, (int(v_x_mid_box), int(v_y_mid_box)), 2, (0, 165, 255), -1
        )  # orange dot for neck

        cv2.rectangle(
            frame,
            (int(xleftT), int(yleftT)),
            (int(xrightB), int(yrightB)),
            (0, 0, 0),
            2,
        )

        # Create mask for the rectangle area
        mask = np.zeros_like(frame)
        cv2.rectangle(
            mask,
            (int(xleftT), int(yleftT)),
            (int(xrightB), int(yrightB)),
            (255, 255, 255),
            -1,
        )  # White rectangle

        # Apply mask to the frame
        masked_frame = cv2.bitwise_and(frame, mask)

        # Find contours in the masked frame
        contours, _ = cv2.findContours(
            masked_frame[:, :, 0], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Find bounding box of non-black pixels with margin
        if contours:
            x, y, w, h = cv2.boundingRect(contours[0])
            x -= crop_margin
            y -= crop_margin
            w += 2 * crop_margin
            h += 2 * crop_margin
            cropped_frame = frame[
                max(0, y) : min(frame.shape[0], y + h),
                max(0, x) : min(frame.shape[1], x + w),
            ]

        else:
            cropped_frame = frame

        # Draw vectors on the cropped frame
        if origin:
            # Draw origin vectors
            cv2.arrowedLine(
                cropped_frame,
                (int(x_origin), int(y_origin)),
                (int(v_x_mid_box), int(v_y_mid_box)),
                (0, 255, 0),
                2,
                tipLength=0.01,
            )  # Red line from neck to nose
            cv2.arrowedLine(
                cropped_frame,
                (int(x_origin), int(y_origin)),
                (int(x_nose_bl6), int(y_nose_bl6)),
                (0, 255, 0),
                2,
                tipLength=0.01,
            )  # Red line from neck to nose
            cv2.arrowedLine(
                cropped_frame,
                (int(x_origin), int(y_origin)),
                (int(x_neck_bl6), int(y_neck_bl6)),
                (0, 255, 0),
                2,
                tipLength=0.01,
            )  # Red line from neck to nose

        # Save the cropped frame
        file_name = f"{experimenter_name}_{mouse_id}_{session}_{experiment_id}_{analyst_name}_{analysis_date}_roi_ang_dir_frame{frameN}.jpg"
        output_path = os.path.join(folder_path, file_name)
        cv2.imwrite(output_path, cropped_frame)
    else:
        print(f"Error reading frame {frameN}")

    # Release video capture object
    vid.release()

    # Close all OpenCV windows
    cv2.destroyAllWindows()

    return output_path


# %%
def display_rectangle_w_control_box(
    video_path, frame_number, xleftT, yleftT, xrightB, yrightB, left
):
    # open video file#
    vid = cv2.VideoCapture(video_path)

    # set frame position
    vid.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    # read frame
    success, frame = vid.read()
    greyFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # check frame read successful
    if success:
        cv2.rectangle(
            greyFrame,
            (int(xleftT), int(yleftT)),
            (int(xrightB), int(yrightB)),
            (0, 255, 0),
            2,
        )
        cv2.rectangle(
            greyFrame,
            (int(left[4, 0]), int(left[4, 1])),
            (int(left[4, 0] + 155), int(left[4, 1] - 155)),
            (0, 255, 0),
            2,
        )
        cv2.rectangle(
            greyFrame,
            (int(left[5, 0]), int(left[5, 1])),
            (int(left[5, 0] - 155), int(left[5, 1] - 155)),
            (0, 255, 0),
            2,
        )
        # display frame
        cv2.imshow("Frame", greyFrame)

        # Loop to wait for key press or window close event
        while True:
            key = cv2.waitKey(1)  # Wait for 1 millisecond
            if key & 0xFF == ord("q"):  # Check if 'q' key is pressed
                break  # Exit the loop if 'q' key is pressed
            if (
                cv2.getWindowProperty("Frame", cv2.WND_PROP_VISIBLE) < 1
            ):  # Check if window is closed
                break  # Exit the loop if window is closed
    else:
        print(f"Error reading frame {frame_number}")

    # release video capture object
    vid.release()

    # close all OpenCV windows
    cv2.destroyAllWindows()


# %%
def fill_df(csv_path):
    """
    fill df with missing frame numbers, for every frame number there needs to be one bl6 and one cd1 row
    """
    df = pd.read_csv(csv_path)

    # Get unique frame numbers for each track
    frame_numbers_bl6 = df[df["track"] == "bl6"]["frame_idx"].unique()
    frame_numbers_cd1 = df[df["track"] == "cd1"]["frame_idx"].unique()

    # Create an array of all possible frame numbers
    all_frame_numbers = np.arange(
        max(frame_numbers_bl6.max(), frame_numbers_cd1.max()) + 1
    )

    # Find missing frame numbers for each track
    missing_frame_numbers_bl6 = np.setdiff1d(all_frame_numbers, frame_numbers_bl6)
    missing_frame_numbers_cd1 = np.setdiff1d(all_frame_numbers, frame_numbers_cd1)

    # Create DataFrame for missing frames for each track
    missing_df_bl6 = pd.DataFrame(
        {"frame_idx": missing_frame_numbers_bl6, "track": "bl6"}
    )
    missing_df_cd1 = pd.DataFrame(
        {"frame_idx": missing_frame_numbers_cd1, "track": "cd1"}
    )

    # Concatenate the original DataFrame with the DataFrames containing missing frames
    df_filled = pd.concat([df, missing_df_bl6, missing_df_cd1])

    # Sort the DataFrame first by 'track' and then by 'frame_idx'
    df_filled = df_filled.sort_values(by=["frame_idx", "track"]).reset_index(drop=True)

    return df, df_filled


# %%
def create_mask_cd1(df, x_bound_left_cd1, x_bound_right_cd1, y_bound_cd1):
    """
    creates mask where either nose or neck of cd1 is outside of bounds
    """
    mask_cd1_nose = np.logical_and(
        df["track"] == "cd1",
        np.logical_or.reduce(
            (
                df["Nose.x"] < x_bound_left_cd1,
                df["Nose.x"] > x_bound_right_cd1,
                df["Nose.y"] > y_bound_cd1,
            )
        ),
    )

    mask_cd1_neck = np.logical_and(
        df["track"] == "cd1",
        np.logical_or.reduce(
            (
                df["Neck.x"] < x_bound_left_cd1,
                df["Neck.x"] > x_bound_right_cd1,
                df["Neck.y"] > y_bound_cd1,
            )
        ),
    )

    # Combine the masks for 'Nose' and 'Neck'
    mask_cd1 = np.logical_or(mask_cd1_nose, mask_cd1_neck)

    return mask_cd1


# %%
def create_mask_bl6(df, x_bound_left_bl6, x_bound_right_bl6, y_bound_bl6):
    """
    creates mask where either nose or neck of bl6 is outside of bounds
    """
    mask_bl6_nose = np.logical_and(
        df["track"] == "bl6",
        np.logical_and.reduce(
            (
                df["Nose.x"] > x_bound_left_bl6,
                df["Nose.x"] < x_bound_right_bl6,
                df["Nose.y"] < y_bound_bl6,
            )
        ),
    )

    mask_bl6_neck = np.logical_and(
        df["track"] == "bl6",
        np.logical_and.reduce(
            (
                df["Neck.x"] > x_bound_left_bl6,
                df["Neck.x"] < x_bound_right_bl6,
                df["Neck.y"] < y_bound_bl6,
            )
        ),
    )

    mask_bl6 = np.logical_or(mask_bl6_nose, mask_bl6_neck)

    return mask_bl6


# %%
def switch_tracks_initial(df_old, merged_mask):
    frames_to_flip = df_old[merged_mask]["frame_idx"]
    df_new = df_old.copy()

    df_switched_frames = df_old[df_old["frame_idx"].isin(frames_to_flip)].copy()

    for frame_number in df_switched_frames["frame_idx"].unique():
        # Get the indices of rows corresponding to the current frame number
        indices = df_switched_frames[
            df_switched_frames["frame_idx"] == frame_number
        ].index

        # Switch track labels for both rows
        for index in indices:
            current_track = df_new.at[index, "track"]
            if current_track == "bl6":
                df_new.at[index, "track"] = "cd1"
            elif current_track == "cd1":
                df_new.at[index, "track"] = "bl6"
    return df_new


# %%
def plot_spatial_switch(
    df1,
    x_column,
    y_column,
    title,
    x_bound_left_cd1,
    x_bound_right_cd1,
    y_bound_cd1,
    x_bound_left_bl6,
    x_bound_right_bl6,
    y_bound_bl6,
    folder_path,
    experimenter_name,
    mouse_id,
    session,
    experiment_id,
    analyst_name,
    analysis_date,
):
    plt.scatter(df1[x_column], df1[y_column], color="black", s=0.1)

    plt.axvline(x=x_bound_left_cd1, color="orange", linestyle="--")
    plt.axvline(x=x_bound_right_cd1, color="orange", linestyle="--")
    plt.axhline(y=y_bound_cd1, color="orange", linestyle="--", label="boundaries cd1")

    plt.axvline(x=x_bound_left_bl6, color="blue", linestyle="--")
    plt.axvline(x=x_bound_right_bl6, color="blue", linestyle="--")
    plt.axhline(y=y_bound_bl6, color="blue", linestyle="--", label="boundaries bl6")

    # Set the limits of the x-axis and y-axis
    plt.xlim(0, 1280)
    plt.ylim(1024, 0)  # Inverted y-axis to match video

    # Add labels and title
    plt.xlabel(f"{x_column} coordinate")
    plt.ylabel(f"{y_column} coordinate")
    plt.title(f"Spatial Plot of {title}")

    # Add legend
    plt.legend()

    # Save the cropped frame
    file_name = f"{experimenter_name}_{mouse_id}_{session}_{experiment_id}_{analyst_name}_{analysis_date}_spatial_plot_of_{title}.svg"
    save_path = os.path.join(folder_path, file_name)

    # Save the plot
    plt.savefig(save_path)

    # Show the plot
    plt.show()


# %%
def clean_after_initial_switch(df, condition):
    columns_to_exclude = ["frame_idx", "track"]
    # Get the indices where the condition is true
    indices_to_assign_nan = df[condition].index
    # Drop the specified columns and assign NaN to the selected rows
    df.loc[indices_to_assign_nan, df.columns.drop(columns_to_exclude)] = np.nan

    return df


# %%
def ambig_frames_bl6(df, y_bound_cd1, x_bound_left_cd1, x_bound_right_cd1):
    frames_bl6_ambig = df[
        (df["track"] == "bl6")
        & (df["Neck.y"] < y_bound_cd1)
        & (df["Neck.x"] > x_bound_left_cd1)
        & (df["Neck.x"] < x_bound_right_cd1)
    ]["frame_idx"].values
    return frames_bl6_ambig


# %%
def ambig_frames_cd1(df, y_bound_cd1, y_bound_bl6, x_bound_left_cd1, x_bound_right_cd1):
    frames_cd1_ambig = df[
        (df["track"] == "cd1")
        & ((df["Neck.y"] > y_bound_bl6) & (df["Neck.y"] < y_bound_cd1))
        & (df["Neck.x"] > x_bound_left_cd1)
        & (df["Neck.x"] < x_bound_right_cd1)
    ]["frame_idx"].values

    frames_cd1_ambig_bl6 = df[
        (
            (df["track"] == "bl6")
            & (df["Neck.y"] < y_bound_cd1)
            & (df["Neck.x"] > x_bound_left_cd1)
            & (df["Neck.x"] < x_bound_right_cd1)
        )
    ]["frame_idx"].values

    indices = np.where(np.isin(frames_cd1_ambig_bl6, frames_cd1_ambig))
    if len(frames_cd1_ambig) > len(frames_cd1_ambig_bl6):
        ambig = frames_cd1_ambig[indices]
    else:
        ambig = frames_cd1_ambig_bl6[indices]
    return ambig


# %%
def vector_angle_diff(df, frames_bl6_ambig):
    v_x_neck_bl6 = df[df["track"] == "bl6"]["Neck.x"]
    v_y_neck_bl6 = df[df["track"] == "bl6"]["Neck.y"]
    v_x_nose_bl6 = df[df["track"] == "bl6"]["Nose.x"]
    v_y_nose_bl6 = df[df["track"] == "bl6"]["Nose.y"]

    x_origin = 0
    y_origin = 0

    # Calculate the vectors
    o_neck_bl6 = np.array([v_x_neck_bl6 - x_origin, v_y_neck_bl6 - y_origin]).T
    o_nose_bl6 = np.array([v_x_nose_bl6 - x_origin, v_y_nose_bl6 - y_origin]).T

    vector_neck_nose_bl6 = o_nose_bl6 - o_neck_bl6

    matrix_v_bl6 = np.hstack(
        (vector_neck_nose_bl6, df["frame_idx"].unique().reshape(-1, 1))
    )

    nan_row = np.full((1, 3), np.nan)

    # Append the NaN row to your matrix
    matrix_v_bl6_b = np.vstack((matrix_v_bl6, nan_row))
    matrix_v_bl6_t = np.vstack((nan_row, matrix_v_bl6))

    matrix_v_bl6_shift = np.hstack((matrix_v_bl6_b, matrix_v_bl6_t))

    matrix_v_bl6_shift = np.delete(
        matrix_v_bl6_shift, 2, axis=1
    )  # remove 3rd column, frame index

    matrix_v_bl6_kill = matrix_v_bl6_shift[1:-1]

    # Extracting the coordinates of the first vector
    x1 = matrix_v_bl6_kill[:, 0]
    y1 = matrix_v_bl6_kill[:, 1]
    # Extracting the coordinates of the second vector
    x2 = matrix_v_bl6_kill[:, 2]
    y2 = matrix_v_bl6_kill[:, 3]
    # Calculate the dot product of the two vectors
    dot_product = x1 * x2 + y1 * y2
    # Calculate the magnitudes of the two vectors
    magnitude1 = np.sqrt(x1**2 + y1**2)
    magnitude2 = np.sqrt(x2**2 + y2**2)
    # Calculate the cosine of the angle between the vectors
    cosine_angle = dot_product / (magnitude1 * magnitude2)
    # Calculate the angle in radians
    angle_radians = np.arccos(cosine_angle)
    # Convert the angle to degrees
    angle_degrees = np.degrees(angle_radians)

    full_matrix = np.hstack((matrix_v_bl6_kill, angle_degrees[:, np.newaxis]))
    diff_angles = abs(np.diff(full_matrix[:, 5]))
    full_matrix_diff_angles = np.hstack(
        (full_matrix[1:, :], diff_angles[:, np.newaxis])
    )

    mask_ambig = np.isin(full_matrix_diff_angles[:, 4], frames_bl6_ambig)
    # Filtered matrix containing rows with frame numbers in the frame_numbers list
    ambig_matrix = full_matrix_diff_angles[mask_ambig]

    return ambig_matrix


# %%
def frames_of_interest(ambig_matrix, cutoff_angle):
    foi_indices_angle = np.where(ambig_matrix[:, 6] >= cutoff_angle)[0]
    foi_indices_nan = np.where(np.isnan(ambig_matrix[:, 6]))[0]
    foi_indices = np.concatenate([foi_indices_angle, foi_indices_nan])
    foi_frames = ambig_matrix[foi_indices, 4]
    return foi_frames


# %%
def get_surrounding_frames(frame_array):
    surrounding_frames = []

    # Iterate over each frame in the array
    for frame in frame_array:
        frame = int(frame)
        # Calculate the indices of the previous and following frames
        prev_frames = range(frame - 7, frame)  # 10
        next_frames = range(frame + 1, frame + 8)  # 11

        # Add the current frame, previous frames, and next frames to the list
        surrounding_frames.extend(prev_frames)
        surrounding_frames.append(frame)
        surrounding_frames.extend(next_frames)

    # Convert the list to a NumPy array, remove duplicates, and sort
    surrounding_frames_array = np.unique(surrounding_frames).astype(int)

    return surrounding_frames_array


# %%
def get_action_array(surrounding_foi_frames_array):
    action_array = np.column_stack(
        (surrounding_foi_frames_array, np.zeros_like(surrounding_foi_frames_array))
    )
    return action_array


# %%
def gui_track_switching(
    video_path,
    action_array,
    surrounding_foi_frames_array,
    df,
    xleftT,
    yleftT,
    xrightB,
    yrightB,
):
    vid = cv2.VideoCapture(video_path)
    current_frame_index = 0
    frame_indices = surrounding_foi_frames_array

    # Flag to indicate if end of video is reached
    end_of_video_reached = False

    while True:
        # Get the current frame number
        frame_number = frame_indices[current_frame_index]

        # Set the video capture object to the specific frame number
        vid.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

        # Read the frame
        ret, frame = vid.read()

        # If frame is read correctly
        if ret:
            # Get the coordinates for the circles
            x_bl6 = df[(df["frame_idx"] == frame_number) & (df["track"] == "bl6")][
                "Neck.x"
            ].values[0]
            y_bl6 = df[(df["frame_idx"] == frame_number) & (df["track"] == "bl6")][
                "Neck.y"
            ].values[0]
            x_cd1 = df[(df["frame_idx"] == frame_number) & (df["track"] == "cd1")][
                "Neck.x"
            ].values[0]
            y_cd1 = df[(df["frame_idx"] == frame_number) & (df["track"] == "cd1")][
                "Neck.y"
            ].values[0]

            # Display the circles on the frame
            if not np.isnan(x_bl6) and not np.isnan(y_bl6):
                cv2.circle(frame, (int(x_bl6), int(y_bl6)), 4, (255, 0, 0), -1)
            if not np.isnan(x_cd1) and not np.isnan(y_cd1):
                cv2.circle(frame, (int(x_cd1), int(y_cd1)), 4, (6, 115, 249), -1)

            # display roi rectangle
            cv2.rectangle(
                frame,
                (int(xleftT), int(yleftT)),
                (int(xrightB), int(yrightB)),
                (255, 255, 255),
                2,
            )

            # Display the frame number if it's not NaN
            if not np.isnan(frame_number):
                cv2.putText(
                    frame,
                    f"Frame: {frame_number}",
                    (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )

            # Display the current value of the array in the top right corner
            current_array_value = action_array[current_frame_index, 1]
            # Determine the corresponding action string based on the current array value
            if current_array_value == 0:
                action_string = "keep"
            elif current_array_value == 1:
                action_string = "switch"
            elif current_array_value == 2:
                action_string = "delete bl6"
            elif current_array_value == 3:
                action_string = "delete cd1"
            elif current_array_value == 4:
                action_string = "delete both"
            else:
                action_string = "unknown"

            # Create the text to display
            text = f"action: {action_string} ({current_array_value})"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
            text_x = (
                frame.shape[1] - text_size[0] - 20
            )  # Adjusted position to ensure text doesn't get cut off

            # Put the text on the frame
            cv2.putText(
                frame, text, (text_x, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2
            )
            # text = f"Array Value: {current_array_value}"
            # text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
            # text_x = frame.shape[1] - text_size[0] - 20  # Adjusted position to ensure text doesn't get cut off
            # cv2.putText(frame, text, (text_x, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Display the frame
            cv2.imshow("Frame", frame)

        # Get the key pressed
        key = cv2.waitKey(0)

        # Check for keyboard input
        if key == ord("a") and current_frame_index > 0:
            current_frame_index = (current_frame_index - 1) % len(frame_indices)
        elif key == ord("d") and current_frame_index < len(frame_indices) - 1:
            current_frame_index = (current_frame_index + 1) % len(frame_indices)
        elif key == ord("s"):
            # Save the frame number for switching
            action_array[current_frame_index, 1] = 1
        elif key == ord("b"):
            # Save the frame number for deleting bl6
            action_array[current_frame_index, 1] = 2
        elif key == ord("c"):
            # Save the frame number for deleting cd1
            action_array[current_frame_index, 1] = 3
        elif key == ord("m"):
            # Save the frame number for deleting both
            action_array[current_frame_index, 1] = 4
        elif key == ord("k"):
            # Save the frame number for keeping
            action_array[current_frame_index, 1] = 0
        elif key == ord("q"):
            break

        # Check if end of video is reached
        if current_frame_index == len(frame_indices) - 1:
            if not end_of_video_reached:
                print("End of video reached.")
                end_of_video_reached = True

        # Wait for a short amount of time to allow the frame to be displayed
        # cv2.waitKey(0)

    # Release video capture and close all windows
    vid.release()
    cv2.destroyAllWindows()


# %%
def apply_actions(action_array, df_old):
    df_new = df_old.copy()
    for frame_number, action_value in zip(action_array[:, 0], action_array[:, 1]):
        # Get the indices of rows corresponding to the current frame number
        indices = df_old[df_old["frame_idx"] == frame_number].index

        # Check if the action value is 1 (indicating switch)
        if action_value == 1:
            # Switch track labels for both rows
            for index in indices:
                current_track = df_new.at[index, "track"]
                if current_track == "bl6":
                    df_new.at[index, "track"] = "cd1"
                elif current_track == "cd1":
                    df_new.at[index, "track"] = "bl6"
        # Check if the action value is 2 (indicating delete bl6)
        if action_value == 2:
            # Iterate over the indices and set entire rows to NaN for 'bl6' track
            for index in indices:
                if df_new.at[index, "track"] == "bl6":
                    # Set entire row to NaN except for 'frame_idx' and 'track' columns
                    df_new.loc[
                        index, df_new.columns.difference(["frame_idx", "track"])
                    ] = np.nan
        # check if action values is 3 (indicating delete cd1)
        if action_value == 3:
            # Iterate over the indices and set entire rows to NaN for 'bl6' track
            for index in indices:
                if df_new.at[index, "track"] == "cd1":
                    # Set entire row to NaN except for 'frame_idx' and 'track' columns
                    df_new.loc[
                        index, df_new.columns.difference(["frame_idx", "track"])
                    ] = np.nan
        if action_value == 4:
            # Iterate over the indices and set entire rows to NaN for 'bl6' track
            for index in indices:
                # Set entire row to NaN except for 'frame_idx' and 'track' columns
                df_new.loc[index, df_new.columns.difference(["frame_idx", "track"])] = (
                    np.nan
                )
    return df_new


# %%
def ambig_frames(frames_bl6_ambig, frames_cd1_ambig):
    set1 = set(frames_bl6_ambig)
    set2 = set(frames_cd1_ambig)

    frames_ambig_combined = list(set1.union(set2))
    return frames_ambig_combined


# %%
def clean_with_gui(
    df,
    y_bound_bl6,
    y_bound_cd1,
    x_bound_left_cd1,
    x_bound_right_cd1,
    cutoff_angle,
    video_path,
    xleftT,
    yleftT,
    xrightB,
    yrightB,
    gui=False,
    last=False,
    surrounding_foi_frames_array_old=None,
    surrounding_foi_frames_array_og=None,
):
    frames_bl6_ambig = ambig_frames_bl6(
        df, y_bound_cd1, x_bound_left_cd1, x_bound_right_cd1
    )

    frames_cd1_ambig = ambig_frames_cd1(
        df, y_bound_cd1, y_bound_bl6, x_bound_left_cd1, x_bound_right_cd1
    )

    frames_ambig_combined = ambig_frames(frames_bl6_ambig, frames_cd1_ambig)

    ambig_matrix = vector_angle_diff(df, frames_ambig_combined)

    foi_frames = frames_of_interest(ambig_matrix, cutoff_angle)

    surrounding_foi_frames_array = get_surrounding_frames(foi_frames)

    if gui:
        surrounding_foi_frames_array = get_surrounding_frames(foi_frames)
        surrounding_foi_frames_array = surrounding_foi_frames_array[
            ~np.isin(surrounding_foi_frames_array, surrounding_foi_frames_array_old)
        ]
        if last:
            surrounding_foi_frames_array = surrounding_foi_frames_array[
                ~np.isin(surrounding_foi_frames_array, surrounding_foi_frames_array_og)
            ]

    action_array = get_action_array(surrounding_foi_frames_array)

    gui_track_switching(
        video_path,
        action_array,
        surrounding_foi_frames_array,
        df,
        xleftT,
        yleftT,
        xrightB,
        yrightB,
    )

    df_new = apply_actions(action_array, df)

    return (
        frames_ambig_combined,
        ambig_matrix,
        surrounding_foi_frames_array,
        action_array,
        df_new,
    )


# %%
