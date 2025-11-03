import kagglehub
import pandas as pd
import os

# Download latest version
path = kagglehub.dataset_download("ahmeduzaki/global-earthquake-tsunami-risk-assessment-dataset")

print("Path to dataset files:", path)

# Load the dataset (adjust filename as needed)
df = pd.read_csv(os.path.join(path, "earthquake_data_tsunami.csv"))

# Save to repo (e.g., in a 'data/raw' folder)
save_path = "./data/raw/earthquake_data_tsunami.csv"
df.to_csv(save_path, index=False)

print("Saved raw dataset to:", save_path)