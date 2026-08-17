import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


# =========================================================
# 1. Project paths
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

input_file = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cleaned_us_housing_market.csv"
)

output_dir = (
    PROJECT_ROOT
    / "data"
    / "forecasts"
    / "regime_aware"
)

output_dir.mkdir(
    parents=True,
    exist_ok=True
)


# =========================================================
# 2. Load data
# =========================================================

df = pd.read_csv(input_file)

df["Date"] = pd.to_datetime(df["Date"])

df = (
    df.sort_values("Date")
      .reset_index(drop=True)
)


print("\n--- Regime-Aware Model: Data Loaded ---")
print("Dataset shape:", df.shape)

print(
    "Date range:",
    df["Date"].min(),
    "to",
    df["Date"].max()
)


# =========================================================
# 3. Create historical features
# =========================================================

df["Time_Index"] = range(len(df))

df["Price_Lag_1"] = (
    df["Median_Price"].shift(1)
)

df["Price_Lag_4"] = (
    df["Median_Price"].shift(4)
)

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


# =========================================================
# 4. Create historical regime indicator
# =========================================================
#
# IMPORTANT:
# The regime is based ONLY on previous-quarter growth.
#
# Therefore we are not using future information.
# =========================================================

def classify_regime(growth):

    if pd.isna(growth):
        return np.nan

    if growth < 0:
        return 0          # Declining

    elif growth < 2:
        return 1          # Stable

    elif growth < 5:
        return 2          # Moderate Growth

    else:
        return 3          # Rapid Growth


df["Market_Regime"] = (
    df["Growth_Lag_1"]
    .apply(classify_regime)
)


# =========================================================
# 5. Human-readable regime labels
# =========================================================

regime_labels = {
    0: "Declining",
    1: "Stable",
    2: "Moderate Growth",
    3: "Rapid Growth"
}


# =========================================================
# 6. Target
# =========================================================

df["Target_Growth"] = (
    df["Median_Price"]
    .pct_change()
    * 100
)


# =========================================================
# 7. Feature set
# =========================================================

features = [
    "Year",
    "Quarter",
    "Time_Index",
    "Price_Lag_1",
    "Price_Lag_4",
    "Growth_Lag_1",
    "Growth_Lag_4",
    "Rolling_Mean_4",
    "Market_Regime"
]


df_model = df[
    ["Date"]
    + features
    + ["Median_Price", "Target_Growth"]
].copy()


df_model = (
    df_model
    .dropna()
    .reset_index(drop=True)
)


print("\n--- Regime-Aware Feature Dataset ---")
print("Shape:", df_model.shape)

print("\nFeatures:")
print(features)


# =========================================================
# 8. Walk-forward validation
# =========================================================

N_FOLDS = 5
TEST_SIZE = 20

results = []

predictions = []


print("\n--- Regime-Aware Walk-Forward Validation ---")

for fold in range(N_FOLDS):

    train_end = (
        len(df_model)
        - TEST_SIZE * (N_FOLDS - fold)
    )

    test_start = train_end

    test_end = (
        test_start
        + TEST_SIZE
    )

    train = df_model.iloc[
        :train_end
    ].copy()

    test = df_model.iloc[
        test_start:test_end
    ].copy()


    print("\n" + "=" * 60)
    print("FOLD", fold + 1)
    print("=" * 60)

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


    # =====================================================
    # Training data
    # =====================================================

    X_train = train[features]

    y_train = train["Target_Growth"]


    X_test = test[features]

    y_test = test["Target_Growth"]


    # =====================================================
    # Train model
    # =====================================================

    model = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        min_samples_leaf=2,
        n_jobs=-1
    )

    model.fit(
        X_train,
        y_train
    )


    # =====================================================
    # Predict growth
    # =====================================================

    predicted_growth = (
        model.predict(X_test)
    )


    # =====================================================
    # Reconstruct prices
    # =====================================================

    previous_prices = (
        test["Price_Lag_1"]
        .values
    )

    predicted_prices = (
        previous_prices
        * (
            1
            + predicted_growth / 100
        )
    )

    actual_prices = (
        test["Median_Price"]
        .values
    )


    # =====================================================
    # Evaluation
    # =====================================================

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
            (
                actual_prices
                - predicted_prices
            )
            / actual_prices
        )
    ) * 100


    print(
        f"MAE : {mae:.2f}"
    )

    print(
        f"RMSE: {rmse:.2f}"
    )

    print(
        f"MAPE: {mape:.2f}%"
    )


    # =====================================================
    # Store fold results
    # =====================================================

    results.append({

        "Fold": fold + 1,

        "MAE": mae,

        "RMSE": rmse,

        "MAPE": mape

    })


    # =====================================================
    # Store predictions
    # =====================================================

    for i in range(len(test)):

        predictions.append({

            "Fold": fold + 1,

            "Date": test["Date"].iloc[i],

            "Actual_Price":
                actual_prices[i],

            "Predicted_Price":
                predicted_prices[i],

            "Actual_Growth":
                test["Target_Growth"].iloc[i],

            "Predicted_Growth":
                predicted_growth[i],

            "Market_Regime":
                regime_labels[
                    int(
                        test[
                            "Market_Regime"
                        ].iloc[i]
                    )
                ]

        })


# =========================================================
# 9. Convert results to DataFrame
# =========================================================

results_df = pd.DataFrame(results)

predictions_df = pd.DataFrame(
    predictions
)


# =========================================================
# 10. Average performance
# =========================================================

mean_mae = (
    results_df["MAE"].mean()
)

mean_rmse = (
    results_df["RMSE"].mean()
)

mean_mape = (
    results_df["MAPE"].mean()
)

std_mape = (
    results_df["MAPE"].std()
)


print("\n" + "=" * 60)
print("REGIME-AWARE MODEL PERFORMANCE")
print("=" * 60)

print(
    f"Mean MAE : {mean_mae:.2f}"
)

print(
    f"Mean RMSE: {mean_rmse:.2f}"
)

print(
    f"Mean MAPE: {mean_mape:.2f}%"
)

print(
    f"MAPE Std : {std_mape:.2f}"
)


# =========================================================
# 11. Performance by regime
# =========================================================

predictions_df["Absolute_Error"] = (
    predictions_df["Actual_Price"]
    - predictions_df["Predicted_Price"]
).abs()


predictions_df["Percentage_Error"] = (
    predictions_df["Absolute_Error"]
    / predictions_df["Actual_Price"]
    * 100
)


regime_performance = (
    predictions_df
    .groupby("Market_Regime")
    .agg(

        Observations=(
            "Actual_Price",
            "count"
        ),

        MAE=(
            "Absolute_Error",
            "mean"
        ),

        MAPE=(
            "Percentage_Error",
            "mean"
        )

    )
    .reset_index()
)


print("\n--- Regime-Aware Performance by Regime ---")

print(
    regime_performance
    .to_string(index=False)
)


# =========================================================
# 12. Save fold results
# =========================================================

fold_file = (
    output_dir
    / "regime_aware_fold_results.csv"
)

results_df.to_csv(
    fold_file,
    index=False
)


# =========================================================
# 13. Save predictions
# =========================================================

prediction_file = (
    output_dir
    / "regime_aware_predictions.csv"
)

predictions_df.to_csv(
    prediction_file,
    index=False
)


# =========================================================
# 14. Save regime performance
# =========================================================

regime_file = (
    output_dir
    / "regime_aware_performance.csv"
)

regime_performance.to_csv(
    regime_file,
    index=False
)


print("\n--- Files Saved ---")

print(
    "Fold results:",
    fold_file
)

print(
    "Predictions:",
    prediction_file
)

print(
    "Regime performance:",
    regime_file
)


# =========================================================
# 15. Research interpretation
# =========================================================

print("\n--- Research Interpretation ---")

print(
    "The regime-aware model explicitly incorporates "
    "historical market-growth conditions into the "
    "forecasting process."
)

print(
    "Its performance should be compared against the "
    "original Random Forest and benchmark models."
)

print(
    "Particular attention should be given to the "
    "Rapid Growth regime, where the original model "
    "showed its highest forecasting error."
)