# %%
import cv2
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import scipy.interpolate
import tkinter as tk
from datetime import datetime
from tkinter import messagebox
from PIL import Image, ImageTk
import statistics
from csv import writer
import pingouin as pg
import pprint
import matplotlib as mpl
import bootstrapped.bootstrap as bs
import bootstrapped.stats_functions as bs_stats

# %%
# Set default label size and title size
# mpl.rcParams["axes.labelsize"] = 12
# mpl.rcParams["axes.titlesize"] = 14
# mpl.rcParams["xtick.labelsize"] = 10
# mpl.rcParams["ytick.labelsize"] = 10

# %%
script_dir = os.path.dirname(os.path.abspath(__file__))
# Set the current working directory to the script directory
os.chdir(script_dir)


# %%
def get_experiment_info_date_only():
    """
    Pop-up window for user-specified experimental information

    Returns:
        experimenter_name
        experiment_id
        analyst_name
        analysis_date
    """
    experiment_info = {}

    def store_input():
        experimenter_name = entry_widgets[0].get()
        experiment_id = entry_widgets[1].get()
        analyst_name = entry_widgets[2].get()
        analysis_date = datetime.now().strftime("%Y-%m-%d")

        # Check if experimenter name and analyst name consist of two initials
        if not (is_initials(experimenter_name) and is_initials(analyst_name)):
            messagebox.showerror(
                "Error", "Name of Experimenter and Analyst must be two initials."
            )
            return

        # Store the input in the experiment_info dictionary
        experiment_info.update(
            {
                "Experimenter Name": experimenter_name,
                "Experiment ID": experiment_id,
                "Analyst Name": analyst_name,
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
        "Experiment ID:",
        "Who is Analysing the Data?",
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


# %% for quicker things
experiment_info = get_experiment_info_date_only()
if experiment_info:
    for key, value in experiment_info.items():
        print(f"{key}: {value}")
else:
    print("Experiment information was not provided.")

experimenter_name = experiment_info["Experimenter Name"]
experiment_id = experiment_info["Experiment ID"]
analyst_name = experiment_info["Analyst Name"]
analysis_date = experiment_info["Analysis Date"]

# %%
folder_path = os.path.join(
    os.getcwd(),
    f"95_ci_5pc_results_target_notarget_{experimenter_name}__{experiment_id}_{analyst_name}_{analysis_date}",
)
# Check if the folder exists, and create it if it doesn't
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

# %%
df_si = pd.read_csv(
    # r"C:\Users\Cystein\Maja\files_td_id\nz_td_id_seconds_info.csv"
    r"D:\5pc_nz_td_id_seconds_info.csv"
)

# %%
df_si = df_si.sort_values(by=["mouse_id", "session"]).reset_index(drop=True)

# %%
df_cohort = pd.read_csv(
    # r"C:\Users\Cystein\Maja\files_td_id\td_id_trial_number_corresponding_mouse_id_subset.csv",
    r"D:\td_id_trial_number_corresponding_mouse_id_subset.csv",
    sep=";",
)

# %%
df_si_cohort = pd.merge(
    df_cohort[["cohort", "mouse_id", "session", "treatment"]],
    df_si,
    on=["mouse_id", "session"],
)

# %%
df_si_cohort["treatment"] = df_si_cohort["treatment"].str.capitalize()


# %%
df_treated = df_si_cohort[df_si_cohort["treatment"] == "Treated"]
df_control = df_si_cohort[df_si_cohort["treatment"] == "Control"]
df_naiv = df_si_cohort[df_si_cohort["treatment"] == "Naiv"]


# %%
df_treated.name = "Treated"
df_control.name = "Control"
df_naiv.name = "Naiv"

# %%
max_s_in_roi = np.nanmax(df_si_cohort["s_in_roi"])
max_s_in_roi_ang_dir = np.nanmax(df_si_cohort["s_in_roi_ang_dir"])


# %%
def pairwise_comparison_sns(
    df1, df2, df3, index, ylab, max_v, ygrid, mult_fact, max_v_bu=None
):
    # Create a figure and axes for the subplots
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(8, 3), sharex=True, sharey=True)

    # Plot for df_naiv
    sns.lineplot(
        data=df1,
        x="session",
        y=index,
        style="mouse_id",
        markers=["o"],
        dashes=False,
        legend=False,
        color="black",
        ax=axes[0],
        markersize=10,
        linewidth=1,
    )
    axes[0].set_xticks([1, 2])
    axes[0].set_xticklabels(["No Target", "Target"])
    axes[0].set_xlabel("")
    axes[0].set_ylabel(ylab, fontsize=12)
    axes[0].set_title(f"{df1.name}", fontsize=14)
    axes[0].grid(axis="y", color=[0.6, 0.6, 0.6])
    axes[0].set_xlim(0.7, 2.3)
    axes[0].set_ylim(-(max_v * 0.05), (max_v + (max_v * 0.1)))
    # axes[0].set_ylim(-(max_v * 0.05), max_v_bu)

    # Plot for df_control
    sns.lineplot(
        data=df2,
        x="session",
        y=index,
        style="mouse_id",
        markers=["o"],
        dashes=False,
        legend=False,
        color="black",
        ax=axes[1],
        markersize=10,
        linewidth=1,
    )
    axes[1].set_xticks([1, 2])
    axes[1].set_xticklabels(["No Target", "Target"])
    axes[1].set_xlabel("Session", fontsize=12)
    axes[1].set_title(f"{df2.name}", fontsize=14)
    axes[1].grid(axis="y", color=[0.6, 0.6, 0.6])
    axes[1].set_ylim(-(max_v * 0.05), (max_v + (max_v * 0.1)))
    # axes[1].set_ylim(-(max_v * 0.05), max_v_bu)

    # Plot for df_treated
    sns.lineplot(
        data=df3,
        x="session",
        y=index,
        style="mouse_id",
        markers=["o"],
        dashes=False,
        legend=False,
        color="black",
        ax=axes[2],
        markersize=10,
        linewidth=1,
    )
    axes[2].set_xticks([1, 2])
    axes[2].set_yticks(np.arange(0, max_v, ygrid))
    axes[2].set_xticklabels(["No Target", "Target"])
    axes[2].set_xlabel("")
    axes[2].set_title(f"{df3.name}", fontsize=14)
    axes[2].grid(axis="y", color=[0.6, 0.6, 0.6])
    axes[2].set_ylim(-(max_v * 0.05), (max_v + (max_v * mult_fact)))
    # axes[2].set_ylim(-(max_v * 0.05), max_v_bu)  # 130 for comparison to button_up plot

    # Adjust layout and show the plot
    plt.tight_layout()
    # plt.subplots_adjust(left=0.1, right=0.3, bottom=0.1, top=0.9, wspace=0.4, hspace=0.4)
    file_name = f"{experimenter_name}_{experiment_id}_{analyst_name}_{analysis_date}_interaction_plot_{index}_naiv_control_treated.svg"
    save_path = os.path.join(folder_path, file_name)

    # Save the plot
    plt.savefig(save_path)
    plt.show()


# %%
pairwise_comparison_sns(
    df_naiv,
    df_control,
    df_treated,
    "s_in_roi",
    "Time in ROI [s]",
    max_s_in_roi,
    ygrid=15,
    mult_fact=0.07,
)
pairwise_comparison_sns(
    df_naiv,
    df_control,
    df_treated,
    "s_in_roi_ang_dir",
    "Time in ROI + Direction [s]",
    max_s_in_roi_ang_dir,
    ygrid=5,
    mult_fact=0.1,
)


# %%
reshaped_df = df_si_cohort.pivot(
    index="mouse_id", columns=["session"], values=["s_in_roi", "s_in_roi_ang_dir"]
)
# add cohort column
df_si_cohort = df_si_cohort.drop_duplicates(subset=["mouse_id"])
reshaped_df["cohort"] = df_si_cohort.set_index("mouse_id")["cohort"]

# %%
reshaped_df["treatment"] = df_si_cohort.groupby("mouse_id")["treatment"].first()
reshaped_df["treatment"] = reshaped_df["treatment"].astype("category")

# %%
reshaped_df["si_index"] = (
    reshaped_df["s_in_roi"][2].values - reshaped_df["s_in_roi"][1].values
) / (reshaped_df["s_in_roi"][2].values + reshaped_df["s_in_roi"][1].values)
reshaped_df["si_ang_dir_index"] = (
    reshaped_df["s_in_roi_ang_dir"][2].values
    - reshaped_df["s_in_roi_ang_dir"][1].values
) / (
    reshaped_df["s_in_roi_ang_dir"][2].values
    + reshaped_df["s_in_roi_ang_dir"][1].values
)

# %%
reshaped_df["si_ratio"] = (reshaped_df["s_in_roi"][2].values) / (
    reshaped_df["s_in_roi"][1].values
)
reshaped_df["si_ratio_ang_dir"] = (reshaped_df["s_in_roi_ang_dir"][2].values) / (
    reshaped_df["s_in_roi_ang_dir"][1].values
)


# %%
def boxplot_index_treatment(df, index, ymin, ymax, label_index):
    sns.boxplot(
        data=df,
        x="treatment",
        y=index,
        order=["Naiv", "Control", "Treated"],
        palette="tab10",
    )

    # Add labels and title
    plt.xlabel("Treatment")
    plt.ylabel(label_index)
    plt.ylim(ymin, ymax)
    plt.axhline(y=0, linestyle="--", color=[0.75, 0.75, 0.75])
    plt.title(f"Boxplot of {label_index}\nby Treatment")

    file_name = f"{experimenter_name}_{experiment_id}_{analyst_name}_{analysis_date}_boxplot_{index}_naiv_control_treated.svg"
    save_path = os.path.join(folder_path, file_name)

    # Save the plot
    plt.savefig(save_path)
    # Show the plot
    plt.show()


# %% boxplot to check for outlier
boxplot_index_treatment(
    reshaped_df, "si_index", -0.5, 1, "Social Interaction Index Time"
)
boxplot_index_treatment(
    reshaped_df,
    "si_ang_dir_index",
    -0.5,
    1,
    "Social Interaction Index Time + Direction",
)
boxplot_index_treatment(
    reshaped_df, "si_ratio", -0.4, 4, "Social Interaction Ratio Time"
)
boxplot_index_treatment(
    reshaped_df,
    "si_ratio_ang_dir",
    -0.4,
    10,
    "Social Interaction Ratio Time + Direction",
)


# %%
# def scatterplot_errorbar(df, index, ymin, ymax, label_index):
#     np.random.seed(7)
#     plt.figure(figsize=(7, 7))
#     mean = df.groupby("treatment")[index].mean()
#     ci_low = df.groupby("treatment")[index].quantile(
#         0.16  # 1SD 68% --> 100-68=32 32/2 = 16
#     )  # 2.5th percentile
#     ci_high = df.groupby("treatment")[index].quantile(
#         0.84  # 1 SD 100-16
#     )  # 97.5th percentile

#     # Get unique treatment labels and assign numeric values
#     treatments = mean.index
#     numeric_treatments = np.arange(len(treatments))

#     sns.stripplot(
#         data=df,
#         x="treatment",
#         y=index,
#         order=["Naiv", "Control", "Treated"],
#         jitter=0.1,
#     )

#     # Plot mean and confidence intervals with jitter
#     plt.errorbar(
#         x=np.array([1, 0, 2]) + 0.25,  # errorbar shifted slightly to the right
#         y=mean.values,
#         yerr=[mean.values - ci_low.values, ci_high.values - mean.values],
#         fmt="o",
#         color="black",
#         capsize=5,
#         capthick=2,
#     )

#     # Add treatment labels
#     plt.xticks(ticks=numeric_treatments)

#     # Add labels and title
#     plt.xlabel("Treatment")
#     plt.ylabel(label_index)
#     plt.title(f"Scatterplot of {label_index}\nby Treatment with Mean and and 68% CI") #

#     plt.ylim(ymin, ymax)
#     plt.axhline(y=0, linestyle="--", color='grey', alpha=0.25)
#     plt.axhline(y=ci_low.values[1], linestyle="--", color="red", alpha=0.4)
#     # plt.axhline(y=ci_high.values[1], linestyle="--", color="red", alpha=0.3)
#     file_name = f"{experimenter_name}_{experiment_id}_{analyst_name}_{analysis_date}_scatterplot_errorbars_{index}_naiv_control_treated.svg"
#     save_path = os.path.join(folder_path, file_name)

#     # Save the plot
#     plt.savefig(save_path)
#     # Show the plot
#     plt.show()
#     return mean, ci_high, ci_low


# %%
def bootstrap_column(df, column_name, quantile):
    # Perform bootstrap analysis on the specified column
    res = df.groupby("treatment")[column_name].apply(
        lambda x: bs.bootstrap(x.values, stat_func=bs_stats.mean, alpha=quantile)
    )

    # Extract BootstrapResults for each treatment
    res_c = res[0]
    res_n = res[1]
    res_t = res[2]

    return res_c, res_n, res_t


# %%
np.random.seed(7)
# TODO change to 0.05 for 90% CI ad 0.025 for 95% CI
quant = 0.025
res_c_si, res_n_si, res_t_si = bootstrap_column(reshaped_df, "si_index", quant)
res_c_si_ang_dir, res_n_si_ang_dir, res_t_si_ang_dir = bootstrap_column(
    reshaped_df, "si_ang_dir_index", quant
)
res_c_si_ratio, res_n_si_ratio, res_t_si_ratio = bootstrap_column(
    reshaped_df, "si_ratio", quant
)
res_c_si_ratio_ang_dir, res_n_si_ratio_ang_dir, res_t_si_ratio_ang_dir = (
    bootstrap_column(reshaped_df, "si_ratio_ang_dir", quant)
)


# %%
def create_series(data, index):
    # Create a pandas Series from the dictionary
    series = pd.Series(data, name=index)

    # Set the name of the index
    series.index.name = "treatment"

    return series


# %%
def generate_results_series(res_c, res_n, res_t, index):
    # Mean data
    mean_data = {"Control": res_c.value, "Naiv": res_n.value, "Treated": res_t.value}
    mean_series = create_series(mean_data, index)

    # Lower confidence interval data
    ci_low_data = {
        "Control": res_c.lower_bound,
        "Naiv": res_n.lower_bound,
        "Treated": res_t.lower_bound,
    }
    ci_low_series = create_series(ci_low_data, index)

    # Upper confidence interval data
    ci_high_data = {
        "Control": res_c.upper_bound,
        "Naiv": res_n.upper_bound,
        "Treated": res_t.upper_bound,
    }
    ci_high_series = create_series(ci_high_data, index)

    return mean_series, ci_low_series, ci_high_series


# %%
mean_series_si, ci_low_series_si, ci_high_series_si = generate_results_series(
    res_c_si, res_n_si, res_t_si, "si_index"
)
mean_series_si_ang_dir, ci_low_series_si_ang_dir, ci_high_series_si_ang_dir = (
    generate_results_series(
        res_c_si_ang_dir, res_n_si_ang_dir, res_t_si_ang_dir, "si_ang_dir_index"
    )
)
mean_series_si_ratio, ci_low_series_si_ratio, ci_high_series_si_ratio = (
    generate_results_series(res_c_si_ratio, res_n_si_ratio, res_t_si_ratio, "si_ratio")
)
(
    mean_series_si_ratio_ang_dir,
    ci_low_series_si_ratio_ang_dir,
    ci_high_series_si_ratio_ang_dir,
) = generate_results_series(
    res_c_si_ratio_ang_dir,
    res_n_si_ratio_ang_dir,
    res_t_si_ratio_ang_dir,
    "si_ratio_ang_dir",
)

# %%
ci = int((1 - (2 * quant)) * 100)


# %%
def scatterplot_errorbar_boot(
    df, index, ymin, ymax, label_index, mean_series, ci_low_series, ci_high_series, ci
):
    np.random.seed(7)
    plt.figure(figsize=(7, 7))

    # Get unique treatment labels and assign numeric values
    treatments = mean_series.index
    numeric_treatments = np.arange(len(treatments))

    sns.stripplot(
        data=df,
        x="treatment",
        y=index,
        order=["Naiv", "Control", "Treated"],
        jitter=0.1,
    )

    # Plot mean and confidence intervals with jitter
    plt.errorbar(
        x=np.array([1, 0, 2]) + 0.25,  # errorbar shifted slightly to the right
        y=mean_series.values,
        yerr=[
            mean_series.values - ci_low_series.values,
            ci_high_series.values - mean_series.values,
        ],
        fmt="o",
        color="black",
        capsize=5,
        capthick=2,
    )

    # Add treatment labels
    plt.xticks(ticks=numeric_treatments)

    # Add labels and title
    plt.xlabel("Treatment")
    plt.ylabel(label_index)
    plt.title(
        f"Scatterplot of {label_index}\nby Treatment with Mean and and {ci}% CI"
    )  #

    plt.ylim(ymin, ymax)
    plt.axhline(y=0, linestyle="--", color="grey", alpha=0.25)
    plt.axhline(y=ci_low_series.values[1], linestyle="--", color="red", alpha=0.4)
    # plt.axhline(y=ci_high.values[1], linestyle="--", color="red", alpha=0.3)
    file_name = f"{experimenter_name}_{experiment_id}_{analyst_name}_{analysis_date}_boot_scatterplot_errorbars_{index}_{ci}_ci_naiv_control_treated.svg"
    save_path = os.path.join(folder_path, file_name)

    # Save the plot
    plt.savefig(save_path)
    # Show the plot
    plt.show()
    return


# %%
# import seaborn as sns
# import matplotlib.pyplot as plt

# np.random.seed(7)
# # Assuming reshaped_df contains your data

# sns.stripplot(
#     data=reshaped_df,
#     x="treatment",
#     y='si_ang_dir_index',
#     order=["naiv", "control", "treated"],
#     jitter=0.1
# )

# # Plotting error bars using sns.pointplot
# ax = sns.pointplot(
#     data=reshaped_df,
#     x="treatment",
#     y='si_ang_dir_index',
#     order=["naiv", "control", "treated"],
#     errorbar=("ci"),  # Choose the type of error bar (e.g., "sd" for standard deviation)
#     capsize=0.1,  # Adjust the size of the caps on the error bars
#     errwidth=1,  # Adjust the width of the error bars
#     alpha=1,  # Adjust the transparency of the error bars
#     color='black',  # Set the color of the error bars
#     markers='o',  # Set markers to empty string to hide the points
#     join=False,
#     n_boot=5000,
#     estimator='mean'
# )

# # Adjust the x position of mean points and error bars together
# shift_amount = 0.25  # Adjusted to shift 0.25 units to the right
# for i in range(len(ax.lines)):
#     ax.lines[i].set_xdata(ax.lines[i].get_xdata() + shift_amount)

# # Access the error bars
# for line in ax.lines:
#     # Check if the line represents an error bar
#     if '_error' in line.get_label():
#         # Extract the y-values of the error bars
#         y_err = line.get_ydata()
#         print("Error bar y-values:", y_err)

# plt.axhline(y=0, linestyle="--", color=[0.65, 0.65, 0.65])
# #plt.axhline(y=ci_low.values[1], linestyle="--", color="red", alpha=0.3)

# plt.show()

# %%
# def scatterplot_errorbar_std(df, index, ymin, ymax):
#     np.random.seed(7)
#     mean = df.groupby("treatment")[index].mean()
#     std_dev = df.groupby("treatment")[index].std()  # Calculate standard deviation

#     # Get unique treatment labels and assign numeric values
#     treatments = mean.index
#     numeric_treatments = np.arange(len(treatments))

#     # Create the scatterplot using Seaborn
#     sns.stripplot(
#         data=df,
#         x="treatment",
#         y=index,
#         order=["naiv", "control", "treated"],
#         jitter=0.1,
#     )

#     # Plot mean and confidence intervals with jitter
#     plt.errorbar(
#         x=np.array([1, 0, 2]) + 0.25,  # errorbar shifted slightly to the right
#         y=mean.values,
#         yerr=std_dev.values,  # Use standard deviation as error bars
#         fmt="o",
#         color="black",
#         capsize=5,
#         capthick=2,
#     )

#     # Add treatment labels
#     plt.xticks(ticks=numeric_treatments)

#     # Add labels and title
#     plt.xlabel("Treatment")
#     plt.ylabel(index)
#     plt.title(f"Scatterplot of {index} by Treatment\nwith Mean and 1 SD Error Bars")

#     plt.ylim(ymin, ymax)
#     plt.axhline(y=0, linestyle="--", color=[0.65, 0.65, 0.65])
#     plt.axhline(y=mean.values[1]-std_dev.values[1], linestyle="--", color="red", alpha=0.3)
#     plt.savefig(f"nz_sus_scatterplot_errorbars_{index}_naiv_control_treated.svg")
#     # Show the plot
#     plt.show()
#     return mean, std_dev  # Return mean and standard deviation


# %%
ymin_si = (np.nanmin(reshaped_df["si_index"].values)) + (
    (np.nanmin(reshaped_df["si_index"].values)) * 0.5
)
ymin_si_ang_dir = (np.nanmin(reshaped_df["si_ang_dir_index"].values)) + (
    (np.nanmin(reshaped_df["si_ang_dir_index"].values)) * 0.5
)

# ymin = np.minimum(ymin_si, ymin_si_ang_dir)
# mean_si, ci_high_si, ci_low_si = scatterplot_errorbar(reshaped_df, "si_index", ymin_si, 1, "Social Interaction Index Time")
# mean_si_ang_dir, ci_high_si_ang_dir, ci_low_si_ang_dir = scatterplot_errorbar(
#     reshaped_df, "si_ang_dir_index", ymin_si_ang_dir, 1, "Soial Interation Index Time + Direction"
# )
# mean_si_ratio, ci_high_si_ratio, ci_low_si_ratio = scatterplot_errorbar(reshaped_df, "si_ratio", -0.2, 4, "Social Interaction Ratio Time")
# # mean_si, std_dev_si = scatterplot_errorbar_std(reshaped_df, "si_index", ymin, 1)
# # mean_si_ang_dir, std_dev_si_ang_dir = scatterplot_errorbar_std(
# #     reshaped_df, "si_ang_dir_index", ymin, 1
# # )
# # mean_si_ratio, std_dev_si_ratio = scatterplot_errorbar_std(reshaped_df, "si_ratio", -0.2, 4)
# mean_si_ratio_ang_dir, ci_high_si_ratio_ang_dir, ci_low_si_ratio_ang_dir = scatterplot_errorbar(reshaped_df, "si_ratio_ang_dir", -0.2, 9, "Social Interaction Ratio Time + Direction")
scatterplot_errorbar_boot(
    reshaped_df,
    "si_index",
    ymin_si,
    1,
    "Soial Interation Index Time",
    mean_series_si,
    ci_low_series_si,
    ci_high_series_si,
    ci,
)
scatterplot_errorbar_boot(
    reshaped_df,
    "si_ang_dir_index",
    ymin_si_ang_dir,
    1,
    "Soial Interation Index Time + Direction",
    mean_series_si_ang_dir,
    ci_low_series_si_ang_dir,
    ci_high_series_si_ang_dir,
    ci,
)
scatterplot_errorbar_boot(
    reshaped_df,
    "si_ratio",
    -0.2,
    4,
    "Soial Interation Ratio Time",
    mean_series_si_ratio,
    ci_low_series_si_ratio,
    ci_high_series_si_ratio,
    ci,
)
scatterplot_errorbar_boot(
    reshaped_df,
    "si_ratio_ang_dir",
    -0.2,
    9,
    "Soial Interation Ratio Time + Direction",
    mean_series_si_ratio_ang_dir,
    ci_low_series_si_ratio_ang_dir,
    ci_high_series_si_ratio_ang_dir,
    ci,
)
# %%
ci_results_file_path = os.path.join(
    folder_path,
    f"{experimenter_name}_{experiment_id}_{analyst_name}_{analysis_date}_{ci}_ci_results.txt",
)

# mean_si_str = pprint.pformat(mean_si)
# ci_low_si_str = pprint.pformat(ci_low_si)
# means_si_ang_dir_str = pprint.pformat(mean_si_ang_dir)
# ci_low_si_ang_dir_str = pprint.pformat(ci_low_si_ang_dir)
# mean_si_ratio_str = pprint.pformat(mean_si_ratio)
# ci_low_si_ratio_str = pprint.pformat(ci_low_si_ratio)
# mean_si_ratio_ang_dir_str = pprint.pformat(mean_si_ratio_ang_dir)
# ci_low_si_ratio_ang_dir_str = pprint.pformat(ci_low_si_ratio_ang_dir)

# sus_threshold_si = ci_low_si.values[1]
# sus_threshold_si_ang_dir = ci_low_si_ang_dir.values[1]
# sus_threshold_si_ratio = ci_low_si_ratio.values[1]
# sus_threshold_si_ratio_ang_dir = ci_low_si_ratio_ang_dir.values[1]
mean_si_str = pprint.pformat(mean_series_si)
ci_low_si_str = pprint.pformat(ci_low_series_si)
means_si_ang_dir_str = pprint.pformat(mean_series_si_ang_dir)
ci_low_si_ang_dir_str = pprint.pformat(ci_low_series_si_ang_dir)
mean_si_ratio_str = pprint.pformat(mean_series_si_ratio)
ci_low_si_ratio_str = pprint.pformat(ci_low_series_si_ratio)
mean_si_ratio_ang_dir_str = pprint.pformat(mean_series_si_ratio_ang_dir)
ci_low_si_ratio_ang_dir_str = pprint.pformat(ci_low_series_si_ratio_ang_dir)

sus_threshold_si = ci_low_series_si.values[1]
sus_threshold_si_ang_dir = ci_low_series_si_ang_dir.values[1]
sus_threshold_si_ratio = ci_low_series_si_ratio.values[1]
sus_threshold_si_ratio_ang_dir = ci_low_series_si_ratio_ang_dir.values[1]

# Write the p-value to the text file
with open(ci_results_file_path, "w") as file:
    file.write(f"Means and low {ci}% CI of Social Interaction Index Time\n")  #
    file.write(mean_si_str)
    file.write("\n\n")
    file.write(ci_low_si_str)
    file.write("\n\n\n")
    file.write(
        f"Means and low {ci}% CI of Social Interaction Index Time + Direction\n"
    )  # low 68% CI
    file.write(means_si_ang_dir_str)
    file.write("\n\n")
    file.write(ci_low_si_ang_dir_str)
    file.write("\n\n\n")
    file.write(
        f"Means and low {ci}% CI of Social Interaction Ratio Time\n"
    )  # low 68% CI
    file.write(mean_si_ratio_str)
    file.write("\n\n")
    file.write(ci_low_si_ratio_str)
    file.write("\n\n\n")
    file.write(
        f"Means and low {ci}% CI of Social Interaction Ratio Time + Direction\n"
    )  # low 68% CI
    file.write(mean_si_ratio_ang_dir_str)
    file.write("\n\n")
    file.write(ci_low_si_ratio_ang_dir_str)
    file.write("\n\n\n")
    file.write(
        f"Animals in the control and treatment groups were classified as susceptible or resilient\nbased on the lower quantile of the {ci}% percentile bootstrap CI of the social interaction index angle and direction of the naiv animals.\n\n"
    )
    file.write(
        f"Threshold to determine susceptible animals based on\nsocial interaction index angle and direction: {sus_threshold_si_ang_dir}\n\n"
    )
    file.write(
        f"Animals in the control and treatment groups with social interaction index angle\nand direction <= {sus_threshold_si_ang_dir} were classified as susceptible.\n\n"
    )
    file.write(
        f"Animals in the control and treatment groups with social interaction index angle\nand direction > {sus_threshold_si_ang_dir} were classified as resilient.\n"
    )
    file.write("\n\n")
    file.write(
        f"Had we used the social interaction ratio time to classify the animals, the threshold would\nhave been {sus_threshold_si_ratio}, "
    )
    file.write(
        f"for social interaction ratio time + direction it would have been {sus_threshold_si_ratio_ang_dir}, "
    )
    file.write(f"and for just the social interaction index time {sus_threshold_si}.")

# %%
# Specify the file path
# file_path = "mean_std_dev_results.txt"

# means_si_str = pprint.pformat(mean_si)
# std_dev_si_str = pprint.pformat(std_dev_si)
# means_si_ang_dir_str = pprint.pformat(mean_si_ang_dir)
# std_dev_si_ang_dir_str = pprint.pformat(std_dev_si_ang_dir)
# mean_si_ratio_str = pprint.pformat(mean_si_ratio)
# std_dev_si_ratio_str = pprint.pformat(std_dev_si_ratio)

# sus_threshold_si = mean_si.values[1]-std_dev_si.values[1]
# sus_threshold_si_ang_dir = mean_si_ang_dir.values[1]-std_dev_si_ang_dir.values[1]
# sus_threshold_si_ratio = mean_si_ratio.values[1]-std_dev_si_ratio.values[1]

# # Write the p-value to the text file
# with open(file_path, "w") as file:
#     file.write(f"Means and SD of social interaction index\n")
#     file.write(means_si_str)
#     file.write("\n\n")
#     file.write(std_dev_si_str)
#     file.write("\n\n\n")
#     file.write(f"Means and SD of social interaction index angle and direction\n")
#     file.write(means_si_ang_dir_str)
#     file.write("\n\n")
#     file.write(std_dev_si_ang_dir_str)
#     file.write("\n\n\n")
#     file.write(f"Means and SD of social interaction ratio\n")
#     file.write(mean_si_ratio_str)
#     file.write("\n\n")
#     file.write(std_dev_si_ratio_str)
#     file.write("\n\n\n")
#     file.write("Animals in the control and treatment groups were classified as susceptible or resilient\nbased on the mean of the social interaction index angle and direction - 1SD of the naiv animals.\n\n")
#     file.write(f"Threshold to determine susceptible animals based on\nsocial interaction index angle and direction: {sus_threshold_si_ang_dir}\n\n")
#     file.write(f"Animals in the control and treatment groups with social interaction index angle\nand direction <= {sus_threshold_si_ang_dir} were classified as susceptible.\n\n")
#     file.write(f"Animals in the control and treatment groups with social interaction index angle\nand direction > {sus_threshold_si_ang_dir} were classified as resilient.\n")
#     file.write("\n\n")
#     file.write(f"Had we used the social interaction ratio to classify the animals, the threshold would\nhave been {sus_threshold_si_ratio}, ")
#     file.write(f"and for just the social interaction index {sus_threshold_si}.")

# %%
np.random.seed(27)
normal_data = np.random.normal(size=len(reshaped_df))


# %%
def density_w_normal_dist(df, index, normal_data, label_index):
    sns.kdeplot(data=df[df["treatment"] == "Naiv"], x=index, label="Naiv")
    sns.kdeplot(
        data=df[df["treatment"] == "Control"],
        x=index,
        label="Control",
    )
    sns.kdeplot(
        data=df[df["treatment"] == "Treated"],
        x=index,
        label="Treated",
    )
    sns.kdeplot(
        data=normal_data, color="black", linestyle="--", label="Normal Distribution"
    )
    plt.legend()
    plt.xlabel(label_index)
    file_name = f"{experimenter_name}_{experiment_id}_{analyst_name}_{analysis_date}_density_plots_w_normal_distribution_{index}_naiv_control_treated.svg"
    save_path = os.path.join(folder_path, file_name)

    # Save the plot
    plt.savefig(save_path)
    plt.show()


# %%
density_w_normal_dist(
    reshaped_df, "si_index", normal_data, "Social Interaction Index Time"
)
density_w_normal_dist(
    reshaped_df,
    "si_ang_dir_index",
    normal_data,
    "Social Interaction Index Time + Direction",
)
density_w_normal_dist(
    reshaped_df, "si_ratio", normal_data, "Social Interaction Ratio Time"
)
density_w_normal_dist(
    reshaped_df,
    "si_ratio_ang_dir",
    normal_data,
    "Social Interaction Ratio Time + Direction",
)


# %%
def density(df, index, label_index):
    plt.figure(figsize=(6.4, 2))
    sns.kdeplot(
        data=df[df["treatment"] == "Control"], x=index, label="Control", linewidth=3
    )
    sns.kdeplot(
        data=df[df["treatment"] == "Treated"], x=index, label="Treated", linewidth=3
    )
    plt.legend()
    plt.xlabel(label_index)
    file_name = f"{experimenter_name}_{experiment_id}_{analyst_name}_{analysis_date}_density_{index}_control_treated.svg"
    save_path = os.path.join(folder_path, file_name)

    # Save the plot
    plt.savefig(save_path)
    plt.show()


# %%
density(reshaped_df, "si_ang_dir_index", "Social Interaction Index Time + Direction")


# %% filter animals into resilient and susceptible
reshaped_df["susceptibility"] = "Naiv"
reshaped_df["susceptibility"] = reshaped_df["susceptibility"].astype("object")

# %%
mask_res = reshaped_df[reshaped_df["treatment"] != "Naiv"]["si_ang_dir_index"] > (
    sus_threshold_si_ang_dir
)

reshaped_df.loc[(reshaped_df["treatment"] != "Naiv") & mask_res, "susceptibility"] = (
    "Resilient"
)

# %%
mask_sus = reshaped_df[reshaped_df["treatment"] != "Naiv"]["si_ang_dir_index"] <= (
    sus_threshold_si_ang_dir
)

reshaped_df.loc[(reshaped_df["treatment"] != "Naiv") & mask_sus, "susceptibility"] = (
    "Susceptible"
)

# %%
# print(reshaped_df)

reshaped_df.to_csv(
    os.path.join(
        folder_path,
        f"{experimenter_name}_{experiment_id}_{analyst_name}_{analysis_date}_td_id_classification_susceptibility_mouse_id.csv",
    )
)


# %%
num_sus_c = len(
    reshaped_df[
        (reshaped_df["treatment"] == "Control")
        & (reshaped_df["susceptibility"] == "Susceptible")
    ]
)
num_total_c = len(reshaped_df[(reshaped_df["treatment"] == "Control")])
p_sus_c = num_sus_c / num_total_c

# %%
num_sus_t = len(
    reshaped_df[
        (reshaped_df["treatment"] == "Treated")
        & (reshaped_df["susceptibility"] == "Susceptible")
    ]
)
num_total_t = len(reshaped_df[(reshaped_df["treatment"] == "Treated")])
# %%
results = scipy.stats.binomtest(
    k=num_sus_t, n=num_total_t, p=p_sus_c, alternative="two-sided"
)

p_val = results.pvalue

# %%
# Specify the file path
binom_results_file_path = os.path.join(
    folder_path,
    f"{experimenter_name}_{experiment_id}_{analyst_name}_{analysis_date}_binomtest_results_{ci}_ci.txt",
)

# Pretty-print the results
results_str = pprint.pformat(results)

# Write the p-value to the text file
with open(binom_results_file_path, "w") as file:
    file.write(
        "Results of the performed binomial test that the probability of success is p\n\n"
    )
    file.write(
        "Null Hypothesis: The probability of success in a Bernoulli Experiment (i.e., the probability of susceptible animals in treatment group) is p.\n"
    )
    file.write("Alternative Hypothesis: The probability of success is not p.\n\n")
    file.write(f"Number of susceptible animals in the control group: {num_sus_c}\n")
    file.write(f"Number of total animals in the control group: {num_total_c}\n")
    file.write(
        f"Probability of success (susceptibility) in control group: {p_sus_c}\n\n"
    )
    file.write(f"Number of susceptible animals in the treatment group: {num_sus_t}\n")
    file.write(f"Number of total animals in the treatment group: {num_total_t}\n\n")
    file.write(results_str)

# %% make open circle marker
pts = np.linspace(0, np.pi * 2, 24)
circ = np.c_[np.sin(pts) / 2, -np.cos(pts) / 2]
vert = np.r_[circ, circ[::-1] * 0.7]
open_circle = mpl.path.Path(vert)


# %%
def sus_col_scatterplot_errorbar_boot(
    df,
    index,
    ymin,
    ymax,
    label_index,
    p_val,
    mean_series,
    ci_low_series,
    ci_high_series,
    ci,
    pos_leg,
):
    np.random.seed(8)
    # plt.figure(figsize=(7, 20))

    # Get unique treatment labels and assign numeric values
    treatments = mean_series.index
    numeric_treatments = np.arange(len(treatments))

    # Define a custom color palette with only blue and red
    custom_palette = sns.color_palette("colorblind")
    custom_palette = [
        custom_palette[0],
        custom_palette[7],
        custom_palette[7],
    ]  # Blue and grey
    # resilient, susceptible, naiv

    # Create the scatterplot using Seaborn
    sns.stripplot(
        data=df[df["susceptibility"] == "Susceptible"],
        x="treatment",
        y=index,
        order=["Naiv", "Control", "Treated"],
        jitter=0.1,
        hue="susceptibility",  # Add hue for susceptibility column
        palette=custom_palette,  # Set colors
        size=12,
        marker=open_circle,  # "$\circ$",
        # edgecolor='black',
        # facecolor='none',
        linewidth=0.001,
        # fill=False
        legend=False,
    )

    sns.stripplot(
        data=df[df["susceptibility"] != "Susceptible"],
        x="treatment",
        y=index,
        order=["Naiv", "Control", "Treated"],
        jitter=0.1,
        hue="susceptibility",  # Add hue for susceptibility column
        palette=custom_palette,  # Set colors
        size=12,
        marker="o",
        edgecolor="white",
        # facecolors='none',
        linewidth=1,
        # fill=False
        legend=False,
    )

    # Plot mean and confidence intervals with jitter
    plt.errorbar(
        x=np.array([1, 0, 2]) + 0.25,  # errorbar shifted slightly to the right
        y=mean_series.values,
        yerr=[
            mean_series.values - ci_low_series.values,
            ci_high_series.values - mean_series.values,
        ],  # Use standard deviation as error bars
        fmt="o",
        color="black",
        capsize=5,
        capthick=3,
        elinewidth=3,
        ms=10,
    )

    # Add treatment labels
    plt.xticks(ticks=numeric_treatments)

    # Add labels and title
    plt.xlabel("Treatment")
    plt.ylabel(label_index)
    plt.title(
        f"Scatterplot of {label_index}\nby Treatment with Mean and {ci}% CI"
    )  # and 68% CI

    plt.ylim(ymin, ymax)
    plt.axhline(y=0, linestyle="--", color="grey", alpha=0.25, linewidth=3)
    plt.axhline(
        y=ci_low_series.values[1], linestyle="--", color="red", alpha=0.4, linewidth=3
    )

    plt.grid(axis="y", color=[0.6, 0.6, 0.6])

    # asterisks = ""
    # if p_val > 0.05:
    #     asterisks = "n.s."
    # if p_val <= 0.05:
    #     asterisks = "*"
    # if p_val <= 0.01:
    #     asterisks = "**"
    # if p_val <= 0.001:
    #     asterisks = "***"
    # if p_val <= 0.0001:
    #     asterisks = "****"
    plt.plot(
        [
            numeric_treatments[1],
            numeric_treatments[1],
            numeric_treatments[2],
            numeric_treatments[2],
        ],
        [0.9, 0.93, 0.93, 0.9],
        lw=1.5,
        c="k",
    )
    # plt.plot(numeric_treatments[1], 1, c='k', marker='|')
    plt.annotate(
        f"$\it{{p}}$ = {p_val}",
        ((numeric_treatments[1] + numeric_treatments[2]) / 2, pos_leg),
        fontsize=12,
        ha="center",
        va="bottom",
    )

    # x_pos = (numeric_treatments[1] + numeric_treatments[2]) / 2
    # plt.annotate(f'p = {p_val}', (x_pos, 1), xytext=(x_pos, 1), fontsize=12,
    #                ha='center', va='bottom', arrowprops=dict(arrowstyle='|-|', lw=1.5))

    # legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label='susceptible',
    #                            markerfacecolor='red', markersize=6)]
    # plt.legend(handles=legend_elements)
    # plt.legend(loc="upper right")
    # this would be the one:
    # legend_elements = [
    # plt.Line2D([0], [0], marker='o', color='w', label='Susceptible',
    #             markerfacecolor='w', markeredgecolor=(0.00392156862745098, 0.45098039215686275, 0.6980392156862745), markersize=5, markeredgewidth=1.5),
    # plt.Line2D([0], [0], marker='o', color='w', label='Resilient',
    #             markerfacecolor=(0.00392156862745098, 0.45098039215686275, 0.6980392156862745), markersize=10),
    # plt.Line2D([0], [0], marker='o', color='w', label='Naiv',
    #             markerfacecolor=(0.5803921568627451, 0.5803921568627451, 0.5803921568627451), markersize=10)
    # ]
    # plt.legend(handles=legend_elements, frameon=False)
    file_name = f"{experimenter_name}_{experiment_id}_{analyst_name}_{analysis_date}_boot_sus_col_scatterplot_errorbars_{index}_{ci}_ci_naiv_control_treated.svg"
    save_path = os.path.join(folder_path, file_name)

    # Save the plot
    plt.savefig(save_path)

    # Show the plot
    plt.show()

    return


# %%
rounded_p_val = np.round(p_val, 3)
formatted_p_val = "{:.3f}".format(rounded_p_val).lstrip("0")  # Remove leading zero
sus_col_scatterplot_errorbar_boot(
    reshaped_df,
    "si_ang_dir_index",
    -0.25,
    1.05,
    "Social Interaction Index Time + Direction",
    formatted_p_val,
    mean_series_si_ang_dir,
    ci_low_series_si_ang_dir,
    ci_high_series_si_ang_dir,
    ci,
    pos_leg=0.94,
)  # 1.05 so legend is not so squished


# %%
df_sus_res = pd.DataFrame(reshaped_df["si_ang_dir_index"])
df_sus_res["treatment"] = reshaped_df["treatment"]
df_sus_res["susceptibility"] = reshaped_df["susceptibility"]
df_sus_res["sus"] = "Naive"
df_sus_res["sus"] = df_sus_res["sus"].astype("object")


# %%
df_sus_res.loc[
    (df_sus_res["treatment"] == "Control")
    & (df_sus_res["susceptibility"] == "Resilient"),
    "sus",
] = "C-Res"

df_sus_res.loc[
    (df_sus_res["treatment"] == "Control")
    & (df_sus_res["susceptibility"] == "Susceptible"),
    "sus",
] = "C-Sus"

df_sus_res.loc[(df_sus_res["treatment"] == "Treated"), "sus"] = "Treated"


# %%
np.random.seed(3003)
# Filter DataFrame for desired categories
categories_to_plot = ["Treated", "C-Sus", "C-Res"]
filtered_df = df_sus_res[df_sus_res["sus"].isin(categories_to_plot)]

# Create boxplot
sns.boxplot(
    data=filtered_df,
    x="sus",
    y="si_ang_dir_index",
    order=categories_to_plot,
    palette="tab10",
    showfliers=False,
)

# Overlay individual points
sns.stripplot(
    data=filtered_df,
    x="sus",
    y="si_ang_dir_index",
    order=categories_to_plot,
    color="black",
    jitter=0.2,
    s=10,
)

# Add labels and title
plt.xlabel("Population")
plt.ylabel("Social Interaction Index")
plt.ylim(-0.5, 1)
plt.axhline(y=0, linestyle="--", color=[0.75, 0.75, 0.75])


file_name = f"{experimenter_name}_{experiment_id}_{analyst_name}_{analysis_date}_boxplot_si_ang_dir_sus_res_control_treated.svg"
save_path = os.path.join(folder_path, file_name)

# Save the plot
plt.savefig(save_path)
# Show the plot
plt.show()


# %%
fig = pg.plot_shift(
    x=reshaped_df[reshaped_df["treatment"] == "Treated"]["si_index"],
    y=reshaped_df[reshaped_df["treatment"] == "Control"]["si_index"],
    paired=False,
    n_boot=10000,
    percentiles=np.array([10, 20, 30, 40, 50, 60, 70, 80, 90]),
    show_median=False,
    seed=7,
    violin=False,
    confidence=0.95,
)

fig.axes[0].set_xlabel("Social Interaction Index Time")
# fig.axes[0].set_xlim(-1, 1)

fig.axes[0].set_yticklabels(["Control", "Treated"])

fig.axes[0].set_title(
    "Comparing Control and Treated -\nSocial Interaction Index Time", size=15
)

fig.axes[1].set_ylabel("Control - Treated\nQuantile Differences (a.u.)")
fig.axes[1].set_xlabel("Treated Quantiles", size=12)
fig.axes[1].set_ylim(-0.6, 0.1)

plt.tight_layout()
file_name = f"{experimenter_name}_{experiment_id}_{analyst_name}_{analysis_date}_shift_function_si.svg"
save_path = os.path.join(folder_path, file_name)

# Save the plot
plt.savefig(save_path)
plt.show()


# %%
fig = pg.plot_shift(
    reshaped_df[reshaped_df["treatment"] == "Treated"]["si_ang_dir_index"],
    reshaped_df[reshaped_df["treatment"] == "Control"][
        "si_ang_dir_index"
    ],  # reshaped_df[(reshaped_df["treatment"] == "Control") & (reshaped_df['susceptibility'] == 'Resilient')]["si_ang_dir_index"],
    paired=False,
    n_boot=10000,
    percentiles=np.array([10, 20, 30, 40, 50, 60, 70, 80, 90]),
    show_median=False,
    seed=7,
    violin=False,
    confidence=0.95,  # TODO set back to 0.68
)

fig.axes[0].set_xlabel("Social Interaction Index Time + Direction")

fig.axes[0].set_yticklabels(["Control", "Treated"])

fig.axes[0].set_title(
    "Comparing Control and Treated -\nSocial Interaction Index Time + Direction",
    size=15,
)

fig.axes[1].set_ylabel("Control - Treated\nQuantile Differences (a.u.)")
fig.axes[1].set_xlabel("Treated Quantiles", size=12)

plt.tight_layout()
file_name = f"{experimenter_name}_{experiment_id}_{analyst_name}_{analysis_date}_shift_function_si_ang_dir_index.svg"
save_path = os.path.join(folder_path, file_name)

# Save the plot
plt.savefig(save_path)
plt.show()


# %%
fig = pg.plot_shift(
    reshaped_df[reshaped_df["treatment"] == "Treated"]["si_ratio"],
    reshaped_df[reshaped_df["treatment"] == "Control"]["si_ratio"],
    paired=False,
    n_boot=10000,
    percentiles=np.array([10, 20, 30, 40, 50, 60, 70, 80, 90]),
    show_median=False,
    seed=7,
    violin=False,
    confidence=0.95,
)

fig.axes[0].set_xlabel("Social Interaction Ratio Time")

fig.axes[0].set_yticklabels(["Control", "Treated"])

fig.axes[0].set_title(
    "Comparing Control and Treated -\nSocial Interaction Ratio Time", size=15
)

fig.axes[1].set_ylabel("Control - Treated\nQuantile Differences (a.u.)")
fig.axes[1].set_xlabel("Treated Quantiles", size=12)

plt.tight_layout()
file_name = f"{experimenter_name}_{experiment_id}_{analyst_name}_{analysis_date}_shift_function_si_ratio.svg"
save_path = os.path.join(folder_path, file_name)

# Save the plot
plt.savefig(save_path)
plt.show()

# %%
fig = pg.plot_shift(
    reshaped_df[reshaped_df["treatment"] == "Treated"]["si_ratio_ang_dir"],
    reshaped_df[reshaped_df["treatment"] == "Control"]["si_ratio_ang_dir"],
    paired=False,
    n_boot=10000,
    percentiles=np.array([10, 20, 30, 40, 50, 60, 70, 80, 90]),
    show_median=False,
    seed=7,
    violin=False,
    confidence=0.95,
)

fig.axes[0].set_xlabel("Social Interaction Ratio Time + Direction")

fig.axes[0].set_yticklabels(["Control", "Treated"])

fig.axes[0].set_title(
    "Comparing Control and Treated -\nSocial Interaction Ratio Time + Direction",
    size=15,
)

fig.axes[1].set_ylabel("Control - Treated\nQuantile Differences (a.u.)")
fig.axes[1].set_xlabel("Treated Quantiles", size=12)

plt.tight_layout()
file_name = f"{experimenter_name}_{experiment_id}_{analyst_name}_{analysis_date}_shift_function_si_ratio_ang_dir.svg"
save_path = os.path.join(folder_path, file_name)

# Save the plot
plt.savefig(save_path)
plt.show()


# %%
# def scatterplot_errorbar_median(df, index, ylim):
#     np.random.seed(7)
#     median = df.groupby("treatment")[index].median()
#     mad = df.groupby("treatment")[index].mad()

#     treatments = median.index
#     numeric_treatments = np.arange(len(treatments))
#     sns.stripplot(
#         data=df,
#         x="treatment",
#         y=index,
#         order=["Naiv", "Control", "Treated"],
#         jitter=0.1,
#     )

#     # Plot median and median absolute deviation with jitter
#     plt.errorbar(
#         x=np.array([1, 0, 2]) + 0.25,  # errorbar shifted slightly to the right
#         y=median.values,
#         yerr=mad.values,
#         fmt="o",
#         color="black",
#         capsize=5,
#         capthick=2,
#     )

#     # Add treatment labels
#     plt.xticks(ticks=numeric_treatments)

#     # Add labels and title
#     plt.xlabel("Treatment")
#     plt.ylabel(index)
#     plt.title(f"Scatterplot of {index} by Treatment\nwith Median and MAD")

#     plt.ylim(ylim, 1)
#     plt.axhline(y=0, linestyle="--", color=[0.65, 0.65, 0.65])
#     plt.axhline(
#         y=(median.values[1] - mad.values[1]), linestyle="--", color="red", alpha=0.3
#     )
#     file_name = f"{experimenter_name}_{experiment_id}_{analyst_name}_{analysis_date}_median_scatterplot_errorbars_{index}_naiv_control_treated.svg"
#     save_path = os.path.join(folder_path, file_name)

#     # Save the plot
#     plt.savefig(save_path)
#     plt.show()

#     return median, mad


# # %%
# median_si, mad_si = scatterplot_errorbar_median(reshaped_df, "si_index", ymin_si)
# median_si_ang_dir, mad_si_ang_dir = scatterplot_errorbar_median(
#     reshaped_df, "si_ang_dir_index", ymin_si_ang_dir
# )

# %%
