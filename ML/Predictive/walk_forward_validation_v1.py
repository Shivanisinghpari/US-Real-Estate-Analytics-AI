"""
US Real Estate Analytics & AI
Walk-Forward Validation + Model Comparison

Purpose:
- Evaluate models using time-series-aware walk-forward validation.
- Avoid relying only on one fixed train/test split.
- Compare:
    1. Seasonal Naive baseline
    2. Random Forest Growth Model
    3. HistGradientBoosting Growth Model
- Report MAE, RMSE, MAPE for each model.
- Save fold-level predictions and final comparison.

This is a research-oriented validation layer: it tests whether the
reported performance is stable across different historical periods.
"""

import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


# =========================================================
# 1. PROJECT PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cleaned_us_housing_market.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "forecasts"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# 2. LOAD DATA
# =========================================================

df = pd.read_csv(INPUT_FILE)

df["Date"] = pd.to_datetime(df["Date"])

df = (
    df.sort_values("Date")
      .reset_index(drop=True)
)

print("\n--- Walk-Forward Validation: Data Loaded ---")
print("Dataset shape:", df.shape)
print(
    "Date range:",
    df["Date"].min(),
    "to",
    df["Date"].max()
)


# =========================================================
# 3. FEATURE ENGINEERING
# =========================================================

df["Time_Index"] = np.arange(len(df))

df["Price_Lag_1"] = df["Median_Price"].shift(1)
df["Price_Lag_4"] = df["Median_Price"].shift(4)

df["Growth_Lag_1"] = (
    df["Median_Price"]
    .pct_change()
    .shift(1)
    * 100
)

df["Growth_Lag_4"] = (
    df["Median_Price"]
    .pct_change(4)
    .shift(1)
    * 100
)

df["Rolling_Mean_4"] = (
    df["Median_Price"]
    .shift(1)
    .rolling(4)
    .mean()
)

df["Target_Growth"] = (
    df["Median_Price"]
    .pct_change()
    * 100
)

FEATURES = [
    "Year",
    "Quarter",
    "Time_Index",
    "Price_Lag_1",
    "Price_Lag_4",
    "Growth_Lag_1",
    "Growth_Lag_4",
    "Rolling_Mean_4",
]

df_model = (
    df[
        ["Date", "Median_Price"] + FEATURES + ["Target_Growth"]
    ]
    .dropna()
    .reset_index(drop=True)
)

print("\n--- Feature Dataset ---")
print("Shape:", df_model.shape)


# =========================================================
# 4. WALK-FORWARD SETTINGS
# =========================================================
#
# We use several historical test windows.
# Each fold trains only on observations before the test window.
#
# This is more appropriate for time-series research than random CV.

N_FOLDS = 5
TEST_SIZE = 20

minimum_training_size = len(df_model) - (N_FOLDS * TEST_SIZE)

if minimum_training_size < 100:
    raise ValueError(
        "Not enough observations for the requested walk-forward setup."
    )


# =========================================================
# 5. METRIC FUNCTION
# =========================================================

def calculate_metrics(actual, predicted):
    actual = np.asarray(actual)
    predicted = np.asarray(predicted)

    mae = mean_absolute_error(actual, predicted)

    rmse = np.sqrt(
        mean_squared_error(actual, predicted)
    )

    # Protect against division by zero.
    non_zero = actual != 0

    if non_zero.sum() == 0:
        mape = np.nan
    else:
        mape = (
            np.mean(
                np.abs(
                    (actual[non_zero] - predicted[non_zero])
                    / actual[non_zero]
                )
            )
            * 100
        )

    return mae, rmse, mape


# =========================================================
# 6. MODEL FACTORIES
# =========================================================

def create_random_forest():
    return RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        min_samples_leaf=2,
        n_jobs=-1
    )


def create_gradient_boosting():
    return HistGradientBoostingRegressor(
        max_iter=300,
        learning_rate=0.05,
        max_leaf_nodes=15,
        l2_regularization=1.0,
        random_state=42
    )


# =========================================================
# 7. WALK-FORWARD VALIDATION
# =========================================================

all_predictions = []
summary_rows = []

n = len(df_model)

print("\n--- Walk-Forward Validation ---")
print("Folds:", N_FOLDS)
print("Test observations per fold:", TEST_SIZE)

for fold in range(N_FOLDS):

    test_end = n - (N_FOLDS - 1 - fold) * TEST_SIZE
    test_start = test_end - TEST_SIZE

    train = df_model.iloc[:test_start].copy()
    test = df_model.iloc[test_start:test_end].copy()

    print(f"\nFold {fold + 1}")
    print(
        "Training:",
        train["Date"].min(),
        "to",
        train["Date"].max()
    )
    print(
        "Testing:",
        test["Date"].min(),
        "to",
        test["Date"].max()
    )

    X_train = train[FEATURES]
    X_test = test[FEATURES]

    y_train = train["Target_Growth"]
    y_test = test["Target_Growth"]

    # -----------------------------
    # Random Forest
    # -----------------------------

    rf = create_random_forest()
    rf.fit(X_train, y_train)

    rf_growth = rf.predict(X_test)

    rf_price = (
        test["Price_Lag_1"].values
        * (1 + rf_growth / 100)
    )

    rf_actual_price = test["Median_Price"].values

    rf_mae, rf_rmse, rf_mape = calculate_metrics(
        rf_actual_price,
        rf_price
    )

    summary_rows.append({
        "Fold": fold + 1,
        "Model": "Random Forest Growth",
        "MAE": rf_mae,
        "RMSE": rf_rmse,
        "MAPE": rf_mape
    })

    fold_rf = pd.DataFrame({
        "Fold": fold + 1,
        "Date": test["Date"].values,
        "Model": "Random Forest Growth",
        "Actual_Price": rf_actual_price,
        "Predicted_Price": rf_price,
        "Actual_Growth": y_test.values,
        "Predicted_Growth": rf_growth
    })

    all_predictions.append(fold_rf)

    # -----------------------------
    # HistGradientBoosting
    # -----------------------------

    gb = create_gradient_boosting()
    gb.fit(X_train, y_train)

    gb_growth = gb.predict(X_test)

    gb_price = (
        test["Price_Lag_1"].values
        * (1 + gb_growth / 100)
    )

    gb_mae, gb_rmse, gb_mape = calculate_metrics(
        rf_actual_price,
        gb_price
    )

    summary_rows.append({
        "Fold": fold + 1,
        "Model": "HistGradientBoosting Growth",
        "MAE": gb_mae,
        "RMSE": gb_rmse,
        "MAPE": gb_mape
    })

    fold_gb = pd.DataFrame({
        "Fold": fold + 1,
        "Date": test["Date"].values,
        "Model": "HistGradientBoosting Growth",
        "Actual_Price": rf_actual_price,
        "Predicted_Price": gb_price,
        "Actual_Growth": y_test.values,
        "Predicted_Growth": gb_growth
    })

    all_predictions.append(fold_gb)

    # -----------------------------
    # Seasonal Naive baseline
    # -----------------------------
    #
    # For quarterly housing data, a simple and important benchmark
    # is the price from four quarters earlier.

    seasonal_naive_price = test["Price_Lag_4"].values

    sn_mae, sn_rmse, sn_mape = calculate_metrics(
        rf_actual_price,
        seasonal_naive_price
    )

    summary_rows.append({
        "Fold": fold + 1,
        "Model": "Seasonal Naive (Lag 4)",
        "MAE": sn_mae,
        "RMSE": sn_rmse,
        "MAPE": sn_mape
    })

    fold_sn = pd.DataFrame({
        "Fold": fold + 1,
        "Date": test["Date"].values,
        "Model": "Seasonal Naive (Lag 4)",
        "Actual_Price": rf_actual_price,
        "Predicted_Price": seasonal_naive_price
    })

    all_predictions.append(fold_sn)


# =========================================================
# 8. SAVE FOLD RESULTS
# =========================================================

fold_results = pd.DataFrame(summary_rows)

fold_results_file = (
    OUTPUT_DIR
    / "walk_forward_fold_results.csv"
)

fold_results.to_csv(
    fold_results_file,
    index=False
)


# =========================================================
# 9. AGGREGATE MODEL PERFORMANCE
# =========================================================

comparison = (
    fold_results
    .groupby("Model", as_index=False)
    .agg({
        "MAE": "mean",
        "RMSE": "mean",
        "MAPE": "mean"
    })
    .sort_values("MAPE")
    .reset_index(drop=True)
)

comparison["MAPE_Std"] = (
    fold_results
    .groupby("Model")["MAPE"]
    .std()
    .reindex(comparison["Model"])
    .values
)

comparison["MAE_Std"] = (
    fold_results
    .groupby("Model")["MAE"]
    .std()
    .reindex(comparison["Model"])
    .values
)

comparison["RMSE_Std"] = (
    fold_results
    .groupby("Model")["RMSE"]
    .std()
    .reindex(comparison["Model"])
    .values
)


# =========================================================
# 10. SELECT BEST MODEL
# =========================================================

best_model = comparison.iloc[0]

print("\n--- Walk-Forward Model Comparison ---")
print(
    comparison.to_string(index=False)
)

print("\n--- Best Walk-Forward Model ---")
print("Model:", best_model["Model"])
print(f"Mean MAE : {best_model['MAE']:.2f}")
print(f"Mean RMSE: {best_model['RMSE']:.2f}")
print(f"Mean MAPE: {best_model['MAPE']:.2f} %")
print(f"MAPE Std : {best_model['MAPE_Std']:.2f}")


# =========================================================
# 11. SAVE COMPARISON
# =========================================================

comparison_file = (
    OUTPUT_DIR
    / "walk_forward_model_comparison.csv"
)

comparison.to_csv(
    comparison_file,
    index=False
)


# =========================================================
# 12. SAVE ALL PREDICTIONS
# =========================================================

predictions = pd.concat(
    all_predictions,
    ignore_index=True
)

predictions_file = (
    OUTPUT_DIR
    / "walk_forward_predictions.csv"
)

predictions.to_csv(
    predictions_file,
    index=False
)


# =========================================================
# 13. FINAL MESSAGE
# =========================================================

print("\n--- Walk-Forward Validation Saved ---")
print("Fold results:")
print(fold_results_file)

print("\nModel comparison:")
print(comparison_file)

print("\nPredictions:")
print(predictions_file)

print(
    "\nResearch note: walk-forward validation evaluates model "
    "stability across multiple historical periods instead of "
    "depending on a single train/test split."
)