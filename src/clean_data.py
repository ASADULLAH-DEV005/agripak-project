import pandas as pd

# Load the data
df = pd.read_csv("data/crop_data.csv")
print("Rows and columns:", df.shape)

# Check for missing values
print("Missing values:")
print(df.isna().sum())

# Check for duplicate rows
print("Duplicate rows:", df.duplicated().sum())

# Remove duplicates and save the cleaned file
df = df.drop_duplicates()
df.to_csv("data/cleaned_data.csv", index=False)
print("Saved data/cleaned_data.csv")