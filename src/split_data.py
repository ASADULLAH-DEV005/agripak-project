import pandas as pd
from sklearn.model_selection import train_test_split

# Load the cleaned data
df = pd.read_csv("data/cleaned_data.csv")

# Split 80% train, 20% test
train, test = train_test_split(
    df,
    test_size=0.2,
    stratify=df["low_yield_risk"],
    random_state=42,
)

# Save the two files
train.to_csv("data/train.csv", index=False)
test.to_csv("data/test.csv", index=False)

print("Train rows:", len(train))
print("Test rows:", len(test))
print("Train low-yield share:", round(train["low_yield_risk"].mean(), 3))
print("Test low-yield share:", round(test["low_yield_risk"].mean(), 3))