import pandas as pd
from pathlib import Path


# ---------------------------------------------------------
# 1. Locate the project root
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

input_file = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ml_features.csv"
)


# ---------------------------------------------------------
# 2. Load ML feature dataset
# ---------------------------------------------------------

df = pd.read_csv(input_file)

df["Date"] = pd.to_datetime(df["Date"])

df = df.sort_values("Date").reset_index(drop=True)


print("\n--- ML Baseline: Data Loaded ---")
print("Dataset shape:", df.shape)

print(
    "Date range:",
    df["Date"].min(),
    "to",
    df["Date"].max()
)


# ---------------------------------------------------------
# 3. Define features and target
# ---------------------------------------------------------

features = [
    "Year",
    "Quarter",
    "Time_Index",
    "Price_Lag_1",
    "Price_Lag_4",
    "Rolling_Mean_4"
]

target = "Median_Price"


X = df[features]
y = df[target]


# ---------------------------------------------------------
# 4. Chronological train/test split
# ---------------------------------------------------------

split_index = int(len(df) * 0.80)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

train_dates = df["Date"].iloc[:split_index]
test_dates = df["Date"].iloc[split_index:]


# ---------------------------------------------------------
# 5. Display split information
# ---------------------------------------------------------

print("\n--- Chronological Train/Test Split ---")

print("Training observations:", len(X_train))
print("Testing observations:", len(X_test))

print("\nTraining period:")
print(train_dates.min(), "to", train_dates.max())

print("\nTesting period:")
print(test_dates.min(), "to", test_dates.max())


print("\n--- Feature Columns ---")
print(features)


from sklearn.ensemble import RandomForestRegressor


# ---------------------------------------------------------
# 6. Train Random Forest Regression model
# ---------------------------------------------------------

model = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    max_depth=None,
    min_samples_leaf=2,
    n_jobs=-1
)

model.fit(X_train, y_train)


print("\n--- Random Forest Model ---")
print("Model training completed.")


# ---------------------------------------------------------
# 7. Generate predictions
# ---------------------------------------------------------

y_pred = model.predict(X_test)


print("\n--- Predictions Generated ---")
print("Number of predictions:", len(y_pred))

print("\nFirst 5 predictions:")

prediction_preview = pd.DataFrame({
    "Date": test_dates.iloc[:5].values,
    "Actual_Price": y_test.iloc[:5].values,
    "Predicted_Price": y_pred[:5]
})

print(prediction_preview)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)
import numpy as np


# ---------------------------------------------------------
# 8. Model Evaluation
# ---------------------------------------------------------

mae = mean_absolute_error(y_test, y_pred)

rmse = np.sqrt(
    mean_squared_error(y_test, y_pred)
)

mape = np.mean(
    np.abs((y_test - y_pred) / y_test)
) * 100


print("\n--- Random Forest Evaluation ---")
print(f"MAE : {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAPE: {mape:.2f} %")

# ---------------------------------------------------------
# 9. Save baseline evaluation results
# ---------------------------------------------------------

results = pd.DataFrame({
    "Model": ["Random Forest Baseline"],
    "MAE": [mae],
    "RMSE": [rmse],
    "MAPE": [mape]
})

results_file = (
    PROJECT_ROOT
    / "data"
    / "forecasts"
    / "ml_baseline_results.csv"
)

results_file.parent.mkdir(
    parents=True,
    exist_ok=True
)

results.to_csv(results_file, index=False)

print("\n--- Results Saved ---")
print(results)

print("\nResults saved to:")
print(results_file)