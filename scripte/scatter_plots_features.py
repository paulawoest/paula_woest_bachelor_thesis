# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os


def plot_scatter_95ci(
    df,
    y,
    estimator,
    title,
    folder_path,
    ymin,
    ymax,
    filename,
    idx_grp1=0,
    idx_grp2=3,
):
    custom_palette = sns.color_palette(
        ["#FFB000", "#1b7837", "#DC267F", "#d9f0d3", "#648FFF"]
    )
    custom_sub_palette = [custom_palette[idx_grp1], custom_palette[idx_grp2]]
    np.random.seed(32)
    g = sns.catplot(
        df,
        x="group",
        y=y,
        kind="strip",
        native_scale=True,
        jitter=0.05,
        hue="group",
        palette=custom_sub_palette,
    )
    sns.pointplot(
        df,
        x="group",
        y=y,
        errorbar=("se"),  # ("ci", 95) statt ("se")
        ax=g.ax,
        linestyle="none",
        capsize=0.05,
        hue="group",
        estimator=estimator,
        palette=custom_sub_palette,
    )
    # Adjust the positions of the pointplot markers
    for i, line in enumerate(g.ax.lines):
        line.set_xdata(line.get_xdata() + 0.15)  # Shift right by 0.15 units
    # adjust position of catplot/stripplot points
    for i, collection in enumerate(g.ax.collections):
        offsets = collection.get_offsets()
        collection.set_offsets(offsets - [0.05, 0])
        # Modify markers: filled for 'sus', unfilled for 'res'
    for collection, group_label in zip(g.ax.collections, df["group"].unique()):
        if "res" in group_label:  # Apply only to 'res' group
            edgecolors = collection.get_facecolors()  # Use the existing colors
            collection.set_facecolors("none")  # Make circles unfilled
            collection.set_edgecolors(edgecolors)  # Set edge colors
            collection.set_linewidth(1.5)  # Adjust linewidth for better visibility
        # Modify markers: filled for 'sus', unfilled for 'res'
    for i, group_label in enumerate(df["group"].unique()):
        if "res" in group_label:  # Apply only to 'res' group mean marker
            mean_marker = g.ax.lines[i]
            mean_marker.set_facecolors("non")  # Make mean marker unfilled
            mean_marker.set_edgecolors(custom_palette[3])  # Set edge color
            mean_marker.set_edgewidth(1.5)
        # Shift right by 0.15 units
    plt.ylim(ymin, ymax)


# %%
folder_path = (
    r"C:\Users\Paula Woest\OneDrive\Desktop\Bachelore Arbeit\Paula\output_data"
)
# %%
# Load the feature matrix CSV file
feature_matrix = pd.read_csv(
    r"C:\Users\Paula Woest\OneDrive\Desktop\Bachelore Arbeit\Paula\output_data\1min_data_feature_matrix_session1.csv"
)

# Extract relevant data (modify column names as necessary)
sus = feature_matrix[feature_matrix["sus_res"] == "sus"][
    "mean_distance_nose_to_tail"
].values[:10]
res = feature_matrix[feature_matrix["sus_res"] == "res"][
    "mean_distance_nose_to_tail"
].values[:9]
# %%
# Create the DataFrame for plotting
data = {
    "mean_distance_nose_to_tail": np.concatenate([sus, res]),
    "group": np.concatenate([np.repeat("sus\nn = 10", 10), np.repeat("res\nn = 9", 9)]),
}
df = pd.DataFrame(data)

# %%
# plot scatterplot with 95% ci mean
plot_scatter_95ci(
    df=df,
    y="mean_distance_nose_to_tail",
    estimator="mean",
    title="Mean Distance",
    folder_path=folder_path,
    ymin=110,
    ymax=180,
    filename="mean_angle_scatterplot_mean_sem_errorbar.svg",
    idx_grp1=1,
    idx_grp2=1,
)
# %%
# %%
import numpy as np
import matplotlib.pyplot as plt

# Set parameters for the two groups
np.random.seed(42)
n = 19
mean_group1 = 10
mean_group2 = 9
sd_group1 = 1
sd_group2 = 1.2
# Generate data for each group
group1 = np.random.normal(mean_group1, sd_group1, n)
group2 = np.random.normal(mean_group2, sd_group2, n)
# Add jitter for x-axis
jitter1 = np.random.uniform(-0.1, 0.1, n)
jitter2 = np.random.uniform(-0.1, 0.1, n)
# Plot the jittered scatterplot
plt.figure(figsize=(8, 5))
plt.scatter(
    np.ones(n) + jitter1,
    group1,
    facecolors="none",
    edgecolors="blue",
    label="Group 1 (Open Circles)",
)
plt.scatter(
    2 * np.ones(n) + jitter2, group2, color="red", label="Group 2 (Filled Circles)"
)
# Customize the plot
plt.xticks([1, 2], ["Category 1", "Category 2"])
plt.ylabel("Values")
plt.title("Jittered Scatterplot of Two Categories")
plt.legend()
plt.grid(alpha=0.3)
plt.show()
# %%
from scipy.stats import sem

# Calculate means and SEM for each group
mean1 = np.mean(sus)
mean2 = np.mean(res)
sem1 = sem(sus)
sem2 = sem(res)
# Plot the jittered scatterplot
plt.figure(figsize=(8, 5))
plt.scatter(
    np.ones(n) + jitter1,
    group1,
    facecolors="none",
    edgecolors="blue",
    label="Group 1 (Open Circles)",
)
plt.scatter(
    2 * np.ones(n) + jitter2, group2, color="red", label="Group 2 (Filled Circles)"
)
# Add mean + SEM as larger circles
plt.scatter(
    1,
    mean1,
    s=100,
    facecolors="none",
    edgecolors="blue",
    linewidth=2,
    label="Group 1 Mean ± SEM",
)
plt.scatter(
    2,
    mean2,
    s=100,
    color="red",
    edgecolors="black",
    linewidth=2,
    label="Group 2 Mean ± SEM",
)
# Error bars for SEM
plt.errorbar(1, mean1, yerr=sem1, fmt="none", ecolor="blue", capsize=5)
plt.errorbar(2, mean2, yerr=sem2, fmt="none", ecolor="red", capsize=5)
# Customize the plot
plt.xticks([1, 2], ["Category 1", "Category 2"])
plt.ylabel("Values")
plt.title("Jittered Scatterplot with Mean ± SEM")
plt.legend()
plt.grid(alpha=0.3)
plt.show()


# %%
def plot_scatter_95ci(
    df,
    y,
    estimator,
    title,
    folder_path,
    ymin,
    ymax,
    filename,
    idx_grp1=0,
    idx_grp2=3,
):
    custom_palette = sns.color_palette(
        ["#FFB000", "#1b7837", "#DC267F", "#785EF0", "#648FFF"]
    )
    custom_sub_palette = [custom_palette[idx_grp1], custom_palette[idx_grp2]]
    np.random.seed(32)

    # Create strip plot with jittered points
    g = sns.catplot(
        df,
        x="group",
        y=y,
        kind="strip",
        native_scale=True,
        jitter=0.05,
        hue="group",
        palette=custom_sub_palette,
    )

    # Add mean and SEM with pointplot
    sns.pointplot(
        df,
        x="group",
        y=y,
        errorbar=("se"),  # Use SEM for error bars
        ax=g.ax,
        linestyle="none",
        capsize=0.05,
        hue="group",
        estimator=estimator,
        palette=custom_sub_palette,
    )

    # Adjust the positions of the pointplot markers
    for i, line in enumerate(g.ax.lines):
        line.set_xdata(line.get_xdata() + 0.15)  # Shift right by 0.15 units

    # Adjust the strip plot points
    for i, collection in enumerate(g.ax.collections):
        offsets = collection.get_offsets()
        collection.set_offsets(offsets - [0.05, 0])  # Shift left by 0.05 units

    # Modify markers: filled for 'sus', unfilled for 'res'
    for i, group_label in enumerate(df["group"].unique()):
        if "res" in group_label:  # Apply only to 'res' group mean marker
            # Get the mean marker for the 'res' group
            mean_marker = g.ax.lines[i]
            mean_marker.set_markerfacecolor("none")  # Make mean marker unfilled
            mean_marker.set_markeredgecolor(custom_sub_palette[1])  # Set edge color
            mean_marker.set_markeredgewidth(1.5)  # Adjust edge width

    plt.ylim(ymin, ymax)
    plt.title(title)

    # Save the plot
    save_path = os.path.join(folder_path, filename)
    plt.savefig(save_path)


# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem

# Example mouse data (replace with your actual data)
mouse_data_sus = feature_matrix[feature_matrix["sus_res"] == "sus"][
    "mean_distance_nose_to_tail"
].values[:10]

mouse_data_res = feature_matrix[feature_matrix["sus_res"] == "res"][
    "mean_distance_nose_to_tail"
].values[:9]

# Calculate group sizes
n_sus = len(mouse_data_sus)
n_res = len(mouse_data_res)

# Add jitter to x-axis positions
jitter_sus = np.random.uniform(-0.1, 0.1, n_sus)
jitter_res = np.random.uniform(-0.1, 0.1, n_res)

# Calculate means and SEM for each group
mean_sus = np.mean(mouse_data_sus)
mean_res = np.mean(mouse_data_res)
sem_sus = sem(mouse_data_sus)
sem_res = sem(mouse_data_res)

# Create the plot
plt.figure(figsize=(8, 5))

# Plot individual data points
plt.scatter(
    np.ones(n_sus) + jitter_sus,
    mouse_data_sus,
    facecolors="none",
    edgecolors="blue",
    label="Susceptible (Open Circles)",
)
plt.scatter(
    2 * np.ones(n_res) + jitter_res,
    mouse_data_res,
    color="red",
    label="Resilient (Filled Circles)",
)

# Plot mean ± SEM as larger circles
plt.scatter(
    1,
    mean_sus,
    s=100,
    facecolors="none",
    edgecolors="blue",
    linewidth=2,
    label="Susceptible Mean ± SEM",
)
plt.scatter(
    2,
    mean_res,
    s=100,
    color="red",
    edgecolors="black",
    linewidth=2,
    label="Resilient Mean ± SEM",
)

# Add SEM error bars
plt.errorbar(1, mean_sus, yerr=sem_sus, fmt="none", ecolor="blue", capsize=5)
plt.errorbar(2, mean_res, yerr=sem_res, fmt="none", ecolor="red", capsize=5)

# Customize the plot
plt.xticks([1, 2], ["Susceptible", "Resilient"])
plt.ylabel("Mean Distance Nose to Tail")
plt.title("Jittered Scatterplot with Mean ± SEM")
plt.legend()
plt.grid(alpha=0.3)

# Show the plot
plt.show()

# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem

# Example mouse data (replace with your actual data)
mouse_data_sus = feature_matrix[feature_matrix["sus_res"] == "sus"][
    "mean_distance_nose_to_tail"
].values[:10]

mouse_data_res = feature_matrix[feature_matrix["sus_res"] == "res"][
    "mean_distance_nose_to_tail"
].values[:9]

# Calculate group sizes
n_sus = len(mouse_data_sus)
n_res = len(mouse_data_res)

# Add jitter to x-axis positions
jitter_sus = np.random.uniform(-0.1, 0.1, n_sus)
jitter_res = np.random.uniform(-0.1, 0.1, n_res)

# Calculate means and SEM for each group
mean_sus = np.mean(mouse_data_sus)
mean_res = np.mean(mouse_data_res)
sem_sus = sem(mouse_data_sus)
sem_res = sem(mouse_data_res)

# Create the plot
plt.figure(figsize=(8, 5))

# Plot individual data points
plt.scatter(
    np.ones(n_sus) + jitter_sus,
    mouse_data_sus,
    color="green",
    alpha=0.8,
    label="Susceptible",
)
plt.scatter(
    2 * np.ones(n_res) + jitter_res,
    mouse_data_res,
    facecolors="none",
    edgecolors="green",
    alpha=0.8,
    label="Resilient",
)

# Plot mean ± SEM
plt.scatter(
    1,
    mean_sus,
    s=100,
    color="green",
    label="Susceptible Mean",
)
plt.scatter(
    2,
    mean_res,
    s=100,
    facecolors="none",
    edgecolors="green",
    linewidth=2,
    label="Resilient Mean",
)

# Add SEM error bars
plt.errorbar(1, mean_sus, yerr=sem_sus, fmt="none", ecolor="green", capsize=5)
plt.errorbar(2, mean_res, yerr=sem_res, fmt="none", ecolor="green", capsize=5)

# Customize the plot
plt.xticks([1, 2], ["sus\nn = 10", "res\nn = 9"])
plt.ylabel("mean_distance_nose_to_tail")
plt.ylim(110, 180)
plt.grid(alpha=0.3)

# Remove legend
plt.legend().set_visible(False)

# Set aspect ratio and show the plot
plt.gca().set_aspect(aspect=1.5)
plt.title("")
plt.tight_layout()
plt.show()

# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem

np.random.seed(4)

# Example mouse data (replace with your actual data)
# Example mouse data (replace with your actual data)
mouse_data_sus = feature_matrix[feature_matrix["sus_res"] == "sus"][
    "mean_distance_nose_to_tail"
].values[:10]

mouse_data_res = feature_matrix[feature_matrix["sus_res"] == "res"][
    "mean_distance_nose_to_tail"
].values[:9]

# Calculate group sizes
n_sus = len(mouse_data_sus)
n_res = len(mouse_data_res)

# Add jitter to x-axis positions
jitter_sus = np.random.uniform(-0.1, 0.1, n_sus)
jitter_res = np.random.uniform(-0.1, 0.1, n_res)

# Calculate means and SEM for each group
mean_sus = np.mean(mouse_data_sus)
mean_res = np.mean(mouse_data_res)
sem_sus = sem(mouse_data_sus)
sem_res = sem(mouse_data_res)

# Create the plot
plt.figure(figsize=(6, 6))  # Increased figure width

# Add SEM error bars
plt.errorbar(
    1.2, mean_sus, yerr=sem_sus, fmt="none", ecolor="green", capsize=5, zorder=0
)
plt.errorbar(
    2.2, mean_res, yerr=sem_res, fmt="white", ecolor="green", capsize=5, zorder=0
)

# Plot individual data points
plt.scatter(
    np.ones(n_sus) + jitter_sus,
    mouse_data_sus,
    color="green",
    alpha=0.8,
    label="Susceptible",
)
plt.scatter(
    2 * np.ones(n_res) + jitter_res,
    mouse_data_res,
    facecolors="none",
    edgecolors="green",
    alpha=0.8,
    label="Resilient",
)

# Plot mean ± SEM
plt.scatter(
    1.2,
    mean_sus,
    s=100,
    color="green",
    label="Susceptible Mean",
)
plt.scatter(
    2.2,
    mean_res,
    s=100,
    facecolors="white",
    edgecolors="green",
    linewidth=2,
    label="Resilient Mean",
    zorder=1,
)

# Customize the plot
plt.xticks([1, 2], ["sus\nn = 10", "res\nn = 9"], fontsize=12)  # Larger x-axis labels
plt.ylabel("mean_distance_nose_to_tail", fontsize=12)
plt.ylim(110, 180)
plt.grid(alpha=0.3)

# Remove legend
plt.legend().set_visible(False)

# Adjust layout for readability
plt.tight_layout()
plt.show()

# %%
