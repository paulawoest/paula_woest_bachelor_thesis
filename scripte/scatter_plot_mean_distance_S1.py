# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem


folder_path = (
    r"C:\Users\Paula Woest\OneDrive\Desktop\Bachelore Arbeit\Paula\output_data"
)
# %%
# Load the feature matrix CSV file
feature_matrix = pd.read_csv(
    r"C:\Users\Paula Woest\OneDrive\Desktop\Bachelore Arbeit\Paula\output_data\Session 1_Single\1min_data_feature_matrix_session1.csv"
)

# Extract relevant data (modify column names as necessary)
sus = feature_matrix[feature_matrix["sus_res"] == "sus"]["mean_distance"].values[:10]
res = feature_matrix[feature_matrix["sus_res"] == "res"]["mean_distance"].values[:9]
# %%
import numpy as np

# Create the DataFrame for plotting
data = {
    "mean_distance": np.concatenate([sus, res]),
    "group": np.concatenate([np.repeat("sus\nn = 10", 10), np.repeat("res\nn = 9", 9)]),
}
df = pd.DataFrame(data)
# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem

np.random.seed(3)

# Example mouse data (replace with your actual data)
# Example mouse data (replace with your actual data)
mouse_data_sus = feature_matrix[feature_matrix["sus_res"] == "sus"][
    "mean_distance"
].values[:10]

mouse_data_res = feature_matrix[feature_matrix["sus_res"] == "res"][
    "mean_distance"
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
    1.2, mean_sus, yerr=sem_sus, fmt="none", ecolor="#762a83", capsize=5, zorder=0
)
plt.errorbar(
    2.2, mean_res, yerr=sem_res, fmt="white", ecolor="#762a83", capsize=5, zorder=0
)

# Plot individual data points
plt.scatter(
    np.ones(n_sus) + jitter_sus,
    mouse_data_sus,
    color="#762a83",
    alpha=0.8,
    label="Susceptible",
)
plt.scatter(
    2 * np.ones(n_res) + jitter_res,
    mouse_data_res,
    facecolors="none",
    edgecolors="#762a83",
    alpha=0.8,
    label="Resilient",
)

# Plot mean ± SEM
plt.scatter(
    1.2,
    mean_sus,
    s=100,
    color="#762a83",
    label="Susceptible Mean",
)
plt.scatter(
    2.2,
    mean_res,
    s=100,
    facecolors="white",
    edgecolors="#762a83",
    linewidth=2,
    label="Resilient Mean",
    zorder=1,
)

# Customize the plot
plt.xticks([1, 2], ["sus\nn = 10", "res\nn = 9"], fontsize=12)  # Larger x-axis labels
plt.ylabel("mean_distance", fontsize=12)
plt.ylim(300, 700)
plt.grid(alpha=0.3)

# Remove legend
plt.legend().set_visible(False)

# Adjust layout for readability
plt.tight_layout()
plt.show()
