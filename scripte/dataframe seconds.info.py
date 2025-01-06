# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# %%
# directory where this script/file is saved
script_dir = os.path.dirname(os.path.abspath(__file__))
# Set the current working directory to the script directory
os.chdir(script_dir)


# %%
import csv

# Specify the path to your CSV file
file_path = (
    r"C:\Users\Paula Woest\OneDrive\Desktop\Bachelore Arbeit\seconds_info_maja.csv"
)


# Open the CSV file
with open(file_path, mode="r", newline="") as csv_file:
    # Create a CSV reader
    csv_reader = csv.reader(csv_file)

    # Read the header (optional)
    header = next(csv_reader)
    print(f"Header: {header}")

    # Read the rows
    for row in csv_reader:
        print(row)
# %%
# Replace 'your_file.csv' with the actual path to your CSV file
df = pd.read_csv(
    r"C:\Users\Paula Woest\OneDrive\Desktop\Bachelore Arbeit\seconds_info_maja.csv"
)

# Display the first few rows to ensure it loaded correctly
print(df.head())
# %%
# Delete rows with NaN values
df = df.dropna()

print("\nDataFrame after removing rows with NaN values:")
print(df)


# %%
# Create a new column based on the existing 'sus_res' column
df["sus_res_numeric"] = df["sus_res"].replace({"sus": 0, "res": 1})

print("\nDataFrame after adding 'sus_res_numeric' using replace():")
print(df)

# %%
df.drop(columns=["sus_res"], inplace=True)

# %%
print(df)
# %%
# import pandas
import pandas as pd

col_names = ["mouse_id", "session", "s_in_roi", "s_in_roi_ang_dir", "sus_res_numeric"]
# %%
# %%df.head()
# split dataset in features and target variable
feature_cols = [
    "mouse_id",
    "session",
    "s_in_roi",
    "s_in_roi_ang_dir",
    "sus_res_numeric",
]
X = df[feature_cols]  # Features
y = df.sus_res_numeric  # Target variable
# %%
# split X and y into training and testing sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=16
)
# %%
# import the class
from sklearn.linear_model import LogisticRegression

# instantiate the model (using the default parameters)
logreg = LogisticRegression(random_state=16)

# fit the model with data
logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)
# %%
# import the metrics class
from sklearn import metrics

cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
cnf_matrix
# %%
# import required modules
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class_names = [0, 1]  # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu", fmt="g")
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title("Confusion matrix", y=1.1)
plt.ylabel("Actual label")
plt.xlabel("Predicted label")

Text(0.5, 257.44, "Predicted label")
# %%
