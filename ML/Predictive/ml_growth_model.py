import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ---------------------------------------------------------
# 1. Locate project files
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

input_file = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cleaned_us_housing_market.csv"
)


# ---------------------------------------------------------
# 2. Load the cleaned housing data
# ---------------------------------------------------------

df = pd.read_csv(input_file)

df["Date"] = pd.to_datetime(df["Date"])

df = (
    df.sort_values("Date")
      .reset_index(drop=True)
)


print("\n--- ML Growth Model: Data Loaded ---")
print("Dataset shape:", df.shape)

print(
    "Date range:",
    df["Date"].min(),
    "to",
    df["Date"].max()
)


# ---------------------------------------------------------
# 3. Create historical features
# ---------------------------------------------------------

df["Time_Index"] = range(len(df))

# Previous-quarter price
df["Price_Lag_1"] = (
    df["Median_Price"].shift(1)
)

# Price four quarters earlier
df["Price_Lag_4"] = (
    df["Median_Price"].shift(4)
)

# Previous-quarter growth
df["Growth_Lag_1"] = (
    df["Median_Price"]
    .pct_change()
    .shift(1)
    * 100
)

# Previous-year growth
df["Growth_Lag_4"] = (
    df["Median_Price"]
    .pct_change(4)
    .shift(1)
    * 100
)

# Historical four-quarter average
df["Rolling_Mean_4"] = (
    df["Median_Price"]
    .shift(1)
    .rolling(4)
    .mean()
)


# ---------------------------------------------------------
# 4. Define the prediction target
# ---------------------------------------------------------

# Growth from the previous quarter to the current quarter.
# The model will only receive information from previous periods.

df["Target_Growth"] = (
    df["Median_Price"]
    .pct_change()
    * 100
)


# ---------------------------------------------------------
# 5. Define ML features
# ---------------------------------------------------------

features = [
    "Year",
    "Quarter",
    "Time_Index",
    "Price_Lag_1",
    "Price_Lag_4",
    "Growth_Lag_1",
    "Growth_Lag_4",
    "Rolling_Mean_4"
]


df_model = df[
    ["Date"] + features + ["Median_Price", "Target_Growth"]
].copy()


# ---------------------------------------------------------
# 6. Remove rows without sufficient history
# ---------------------------------------------------------

df_model = (
    df_model
    .dropna()
    .reset_index(drop=True)
)


print("\n--- Growth Modeling Dataset ---")
print("Shape:", df_model.shape)

print("\n--- Features ---")
print(features)

print("\n--- Missing Values ---")
print(df_model.isnull().sum())


# ---------------------------------------------------------
# 7. Chronological train/test split
# ---------------------------------------------------------

split_index = int(len(df_model) * 0.80)

train = df_model.iloc[:split_index].copy()
test = df_model.iloc[split_index:].copy()


print("\n--- Chronological Train/Test Split ---")

print("Training observations:", len(train))
print("Testing observations:", len(test))

print("\nTraining period:")
print(
    train["Date"].min(),
    "to",
    train["Date"].max()
)

print("\nTesting period:")
print(
    test["Date"].min(),
    "to",
    test["Date"].max()
)


# ---------------------------------------------------------
# 8. Prepare training and testing data
# ---------------------------------------------------------

X_train = train[features]
X_test = test[features]

y_train = train["Target_Growth"]
y_test = test["Target_Growth"]


# ---------------------------------------------------------
# 9. Train Random Forest growth model
# ---------------------------------------------------------

model = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    min_samples_leaf=2,
    n_jobs=-1
)

model.fit(X_train, y_train)


print("\n--- Random Forest Growth Model ---")
print("Model training completed.")


# ---------------------------------------------------------
# 10. Predict quarterly growth
# ---------------------------------------------------------

predicted_growth = model.predict(X_test)


print("\n--- Growth Predictions ---")

growth_preview = pd.DataFrame({
    "Date": test["Date"].iloc[:5].values,
    "Actual_Growth": y_test.iloc[:5].values,
    "Predicted_Growth": predicted_growth[:5]
})

print(growth_preview)

# ---------------------------------------------------------
# 11. Reconstruct predicted house prices
# ---------------------------------------------------------

previous_prices = test["Price_Lag_1"].values

predicted_prices = (
    previous_prices
    * (1 + predicted_growth / 100)
)


actual_prices = test["Median_Price"].values


# ---------------------------------------------------------
# 12. Display reconstructed prices
# ---------------------------------------------------------

price_preview = pd.DataFrame({
    "Date": test["Date"].values[:5],
    "Actual_Price": actual_prices[:5],
    "Predicted_Price": predicted_prices[:5],
    "Actual_Growth": y_test.values[:5],
    "Predicted_Growth": predicted_growth[:5]
})


print("\n--- Reconstructed Price Predictions ---")
print(price_preview)

# ---------------------------------------------------------
# 13. Evaluate reconstructed price predictions
# ---------------------------------------------------------

mae = mean_absolute_error(
    actual_prices,
    predicted_prices
)

rmse = np.sqrt(
    mean_squared_error(
        actual_prices,
        predicted_prices
    )
)

mape = np.mean(
    np.abs(
        (actual_prices - predicted_prices)
        / actual_prices
    )
) * 100


print("\n--- Random Forest Growth Model Evaluation ---")
print(f"MAE : {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAPE: {mape:.2f} %")

# ---------------------------------------------------------
# 14. Save growth model evaluation results
# ---------------------------------------------------------

results = pd.DataFrame({
    "Model": ["Random Forest Growth Model"],
    "MAE": [mae],
    "RMSE": [rmse],
    "MAPE": [mape]
})

results_file = (
    PROJECT_ROOT
    / "data"
    / "forecasts"
    / "ml_growth_model_results.csv"
)

results_file.parent.mkdir(
    parents=True,
    exist_ok=True
)

results.to_csv(results_file, index=False)

print("\n--- Growth Model Results Saved ---")
print(results)

print("\nResults saved to:")
print(results_file)

# ---------------------------------------------------------
# 15. Feature Importance Analysis
# ---------------------------------------------------------

feature_importance = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
})

feature_importance = (
    feature_importance
    .sort_values(
        "Importance",
        ascending=False
    )
    .reset_index(drop=True)
)


print("\n--- Random Forest Feature Importance ---")
print(feature_importance)


# ---------------------------------------------------------
# 16. Save feature importance
# ---------------------------------------------------------

importance_file = (
    PROJECT_ROOT
    / "data"
    / "forecasts"
    / "ml_growth_feature_importance.csv"
)

feature_importance.to_csv(
    importance_file,
    index=False
)


print("\nFeature importance saved to:")
print(importance_file)

# ---------------------------------------------------------
# 17. Actual vs Predicted Price Visualization
# ---------------------------------------------------------

import matplotlib.pyplot as plt


plt.figure(figsize=(12, 6))

plt.plot(
    test["Date"],
    actual_prices,
    label="Actual Price"
)

plt.plot(
    test["Date"],
    predicted_prices,
    label="Predicted Price"
)

plt.xlabel("Date")
plt.ylabel("Median House Price")
plt.title(
    "Random Forest Growth Model: Actual vs Predicted Prices"
)

plt.legend()
plt.grid(True)
plt.tight_layout()


# ---------------------------------------------------------
# 18. Save visualization
# ---------------------------------------------------------

plot_file = (
    PROJECT_ROOT
    / "data"
    / "forecasts"
    / "ml_growth_actual_vs_predicted.png"
)

plt.savefig(
    plot_file,
    dpi=300,
    bbox_inches="tight"
)

plt.show()


print("\nVisualization saved to:")
print(plot_file)