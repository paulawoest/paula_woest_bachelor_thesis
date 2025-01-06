# %%
# import cv2
import numpy as np
import os
from csv import writer
import cv2
import statistics

# import analysis functions
import td_id_analysis_sleap_svenja_functions as sleapf

# %%
# directory where this script/file is saved
script_dir = os.path.dirname(os.path.abspath(__file__))
# Set the current working directory to the script directory
os.chdir(script_dir)
# set folder path
folder_path = os.getcwd()


# %%
def display_arena_set_image(image_path):
    """
    Display an image and allow users to click top left and bottom right corners of a box to determine ROI.
    """
    # Read the image
    frame = cv2.imread(image_path)

    # Check if image is successfully read
    if frame is not None:
        # Create a copy of the frame to draw on
        frame_copy = frame.copy()

        # Display the image
        cv2.imshow("Image", frame_copy)

        # Create a list to store coordinates of the corners
        corners = []

        def click_event(event, x, y, flags, params):
            """
            Function to record x- and y-coordinates of left mouse-click
            """
            nonlocal corners

            if event == cv2.EVENT_LBUTTONDOWN:
                # Draw a red circle at the clicked point
                cv2.circle(frame_copy, (x, y), 4, (0, 0, 255), -1)
                cv2.imshow(
                    "Image", frame_copy
                )  # Update the displayed image with the drawn circle
                corners.append((x, y))
                print(x, y)

                # Check if the desired number of clicks is reached
                if len(corners) == 2:
                    # Wait for 2ms
                    cv2.waitKey(500)
                    # Close the window
                    cv2.destroyAllWindows()

        # Set mouse callback function
        cv2.setMouseCallback("Image", click_event)

        # Wait for key press to exit
        cv2.waitKey(0)

        # Close the window
        cv2.destroyAllWindows()

        return corners

    else:
        print(f"Error reading image: {image_path}")
        return None


# %%
def display_rectangle_save_roi(
    mouse_id, video_path, frame_number, xleftT, yleftT, xrightB, yrightB
):
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

        # save the modified frame
        save_path = f"SB_{mouse_id}_session2_uSoIn_PW_roi_coh_{cohort}.jpg"
        cv2.imwrite(save_path, greyFrame)

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
# set cohort
cohort = int(input("Enter cohort number: "))  # i.e. either 6 or 7
# set exemplary mouse id
mouse_id = str(input("Enter exemplary mouse ID: "))
# set path to ethovision arena image
arena_settings_img_path = rf"C:\Users\Cystein\Paula\Arena_Map_C{cohort}.jpg"
# example video to display roi onto
video_path = os.path.join(
    folder_path,
    f"{mouse_id}",
    "session2",
    f"grayscale_coh_{cohort}_{mouse_id}_session2.mp4",
)
vid = cv2.VideoCapture(video_path)
total_frame_count = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
fps = vid.get(cv2.CAP_PROP_FPS)
# frame number you want to display --> 0 so that box is not obstructed by bl6 mouse
frame_number = 0
# labelled instruction image that pops up before corners of box are clicked
image_path = r"C:\Users\Cystein\Maja\SLEAP\files_td_id\frame3_labelled_example.jpg"

# %% #TODO get boundaries of box
sleapf.display_instructions(image_path)
# display the frame and select corners for rectangle to be drawn
coordinates_box = sleapf.display_frame(video_path, frame_number)
#  get boundaries of box
boundaries_box = np.array(coordinates_box["left"], dtype=np.float64)

# %% set boundaries for cd1 mouse, i.e., cd1 mouse cannot be outside of box
x_bound_left_cd1 = (
    int(boundaries_box[0, 0]) - 5
)  # - 5 to account for slight inaccuracies
x_bound_right_cd1 = (
    int(boundaries_box[2, 0]) + 5
)  # + 5 to account for slight inaccuracies
y_bound_cd1 = int(
    statistics.mean((int(boundaries_box[1, 1]), int(boundaries_box[3, 1])))
)

# %% set boundaries for bl6 mouse, i.e., bl6 mouse cannot be inside of box
x_bound_left_bl6 = (
    int(boundaries_box[0, 0]) + 40
)  # + 40 because of perspective, we cannot use exact boundaries of box, but more strict ones
x_bound_right_bl6 = (
    int(boundaries_box[2, 0]) - 40
)  # - 40 because of perspective, we cannot use exact boundaries of box, but more strict ones
y_bound_bl6 = (
    int(statistics.mean((int(boundaries_box[0, 1]), int(boundaries_box[2, 1])))) - 10
)  # -10 to account for slight tracking inaccuracies


# %% #TODO get boundaries roi arena
corners_roi = display_arena_set_image(arena_settings_img_path)

# %% get boundaries of ROI
boundaries_roi = np.array(corners_roi, dtype=np.float64)
xleftT = boundaries_roi[0, 0]
yleftT = boundaries_roi[0, 1]
xrightB = boundaries_roi[1, 0]
yrightB = boundaries_roi[1, 1]

# %%
display_rectangle_save_roi(
    mouse_id, video_path, frame_number, xleftT, yleftT, xrightB, yrightB
)

# %% save boundary/roi coordinates to csv files
# List that we want to add as a new row
header = [
    "cohort",
    "mouse_id",
    "fps",
    "total_frame_count",
    "x_bound_left_cd1",
    "x_bound_right_cd1",
    "y_bound_cd1",
    "x_bound_left_bl6",
    "x_bound_right_bl6",
    "y_bound_bl6",
    "xleftT",
    "yleftT",
    "xrightB",
    "yrightB",
]
video_info = [
    cohort,
    mouse_id,
    fps,
    total_frame_count,
    x_bound_left_cd1,
    x_bound_right_cd1,
    y_bound_cd1,
    x_bound_left_bl6,
    x_bound_right_bl6,
    y_bound_bl6,
    xleftT,
    yleftT,
    xrightB,
    yrightB,
]

# Check if the CSV file exists
file_exists = os.path.isfile("video_info.csv")

# Open our existing CSV file in append mode
# Create a file object for this file
# nz for new zone analysis
with open("video_info.csv", "a") as f_object:  # a is for append

    # Pass this file object to csv.writer() and get a writer object
    writer_object = writer(f_object)

    # Write the header row if the file doesn't exist
    if not file_exists:
        writer_object.writerow(header)

    # Write the data row
    writer_object.writerow(video_info)

# %%
