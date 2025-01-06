# %%
import pandas as pd
import matplotlib.pyplot as plt
import random

# Step 1: Load the feature matrix (assuming it's a CSV file for this example)
feature_matrix = pd.read_csv(
    r"C:\Users\Paula Woest\OneDrive\Desktop\Bachelore Arbeit\Paula\output_data\Session 1_Single\1min_data_feature_matrix_session1.csv"
)

# Step 2: Plot the data
# Separate data by category
susceptible = feature_matrix[feature_matrix["sus_res"] == "sus"]
resilient = feature_matrix[feature_matrix["sus_res"] == "res"]

plt.figure(figsize=(6, 6))

# Plot susceptible mice (blue)
plt.scatter(
    susceptible["mean_location__Neck.x"],
    susceptible["mean_location__Neck.y"],
    color="#762a83",
    label="susceptible",
    alpha=0.8,
)

# Plot resilient mice (orange)
plt.scatter(
    resilient["mean_location__Neck.x"],
    resilient["mean_location__Neck.y"],
    facecolor="none",
    edgecolor="#762a83",
    label="resilient",
    alpha=0.8,
)

# Add labels and title
plt.xlabel("X Coordinate", fontsize=13)
plt.ylabel("Y Coordinate", fontsize=13)


# Add legend
plt.legend(bbox_to_anchor=(1.32, 0.111))

# Add grid
plt.grid(False)

# Show plot
plt.show()


# %%
