import pandas as pd

# Load the cleaned data
df = pd.read_csv("data/cleaned_data.csv")

# Turn word columns into 0/1 columns so we can calculate correlation
encoded = pd.get_dummies(df, dtype=int)

# Correlation of every column with the target
corr = encoded.corr()["low_yield_risk"].drop("low_yield_risk")

# Show all columns, strongest first (ignoring + or - sign)
ordered = corr.abs().sort_values(ascending=False)
print("Correlation with low_yield_risk (absolute value):")
print(ordered.round(3).to_string())

# Flag anything above 0.9
flagged = ordered[ordered > 0.9]
print()
if flagged.empty:
    print("No column is correlated above 0.9.")
else:
    print("FLAGGED columns:")
    print(flagged.round(3).to_string())