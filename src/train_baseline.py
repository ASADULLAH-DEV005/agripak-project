import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Load the train and test files
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

# Separate the inputs (X) from the answer we want to predict (y)
X_train = train.drop(columns="low_yield_risk")
y_train = train["low_yield_risk"]
X_test = test.drop(columns="low_yield_risk")
y_test = test["low_yield_risk"]

# Word columns and number columns need different preparation
word_cols = ["region", "soil_type", "crop", "weather_condition"]
number_cols = ["rainfall_mm", "temperature_c", "fertilizer_used",
               "irrigation_used", "days_to_harvest"]

prepare = ColumnTransformer([
    ("numbers", StandardScaler(), number_cols),
    ("words", OneHotEncoder(handle_unknown="ignore"), word_cols),
])

model = Pipeline([
    ("prepare", prepare),
    ("classifier", LogisticRegression(max_iter=1000)),
])

# Learn from the train data only
model.fit(X_train, y_train)

# Test on data the model has never seen
predictions = model.predict(X_test)

print("Accuracy :", round(accuracy_score(y_test, predictions), 4))
print("Precision:", round(precision_score(y_test, predictions), 4))
print("Recall   :", round(recall_score(y_test, predictions), 4))
print("F1 score :", round(f1_score(y_test, predictions), 4))