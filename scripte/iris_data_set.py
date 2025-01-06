# %%
from sklearn.datasets import load_iris
import pandas as pd
import os

# Load the Iris dataset
iris = load_iris()
data = pd.DataFrame(iris.data, columns=iris.feature_names)
# Add the species column
data["species"] = iris.target
# Map species numbers to names
species_mapping = {0: "setosa", 1: "versicolor", 2: "virginica"}
data["species"] = data["species"].map(species_mapping)
# Keep only two species (setosa and versicolor) and omit the third (virginica)
filtered_data = data[data["species"].isin(["setosa", "versicolor"])]
# Rename the species column to 'sus_res'
filtered_data = filtered_data.rename(columns={"species": "sus_res"})
# Display the first few rows of the filtered dataset
print(filtered_data.head())
# If you want to save the dataset to a CSV file
filtered_data.to_csv("iris_two_species.csv", index=False)
# %%
from sklearn.datasets import load_iris
import pandas as pd
import os

# Load the Iris dataset
iris = load_iris()
data = pd.DataFrame(iris.data, columns=iris.feature_names)

# Add the species column
data["species"] = iris.target

# Map species numbers to names
species_mapping = {0: "setosa", 1: "versicolor", 2: "virginica"}
data["species"] = data["species"].map(species_mapping)

# Keep only two species (setosa and versicolor)
filtered_data = data[data["species"].isin(["setosa", "versicolor"])]

# Rename the species column to 'sus_res'
filtered_data = filtered_data.rename(columns={"species": "sus_res"})

# Select the first 10 rows for each species
filtered_data = filtered_data.groupby("sus_res").head(10).reset_index(drop=True)

# Display the first few rows of the filtered dataset
print(filtered_data)

# Save the dataset to a CSV file
# filtered_data.to_csv("iris_two_species_10_rows.csv", index=False)


# %%
root_path = r"C:\Users\Paula Woest\OneDrive\Desktop\Bachelore Arbeit\Paula"
folder_path = os.path.join(root_path, "output_data", "reduced_irisdataset.csv")
# %%
filtered_data.to_csv(folder_path, index=False)
# %%
