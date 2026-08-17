import pandas as pd
import numpy as np
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

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "forecasts"
)

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

print("\n--- Benchmark Experiment: Data Loaded ---")
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
    "Rolling_Mean_4"
]


df_model = df[
    ["Date"]
    + FEATURES
    + ["Median_Price", "Target_Growth"]
].dropna().reset_index(drop=True)


print("\n--- Benchmark Feature Dataset ---")
print("Shape:", df_model.shape)


# =========================================================
# 4. WALK-FORWARD SETTINGS
# =========================================================

N_FOLDS = 5
TEST_SIZE = 20

print("\n--- Walk-Forward Benchmark Validation ---")
print("Folds:", N_FOLDS)
print("Test observations per fold:", TEST_SIZE)


# =========================================================
# 5. MODEL FUNCTIONS
# =========================================================

def evaluate_predictions(actual_prices, predicted_prices):

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

    return mae, rmse, mape


def reconstruct_prices(previous_prices, predicted_growth):

    return (
        previous_prices
        * (1 + predicted_growth / 100)
    )


# =========================================================
# 6. WALK-FORWARD VALIDATION
# =========================================================

fold_results = []

n = len(df_model)

initial_train_size = (
    n - N_FOLDS * TEST_SIZE
)


for fold in range(1, N_FOLDS + 1):

    train_end = (
        initial_train_size
        + (fold - 1) * TEST_SIZE
    )

    test_end = (
        train_end
        + TEST_SIZE
    )

    train = df_model.iloc[:train_end].copy()

    test = df_model.iloc[
        train_end:test_end
    ].copy()

    print("\n" + "=" * 60)
    print(f"FOLD {fold}")
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


    X_train = train[FEATURES]
    y_train = train["Target_Growth"]

    X_test = test[FEATURES]

    actual_prices = (
        test["Median_Price"].values
    )

    previous_prices = (
        test["Price_Lag_1"].values
    )


    # =====================================================
    # MODEL 1: RANDOM FOREST
    # =====================================================

    rf = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        min_samples_leaf=2,
        n_jobs=-1
    )

    rf.fit(
        X_train,
        y_train
    )

    rf_growth = rf.predict(
        X_test
    )

    rf_prices = reconstruct_prices(
        previous_prices,
        rf_growth
    )

    rf_mae, rf_rmse, rf_mape = (
        evaluate_predictions(
            actual_prices,
            rf_prices
        )
    )


    # =====================================================
    # MODEL 2: HISTORICAL MEAN GROWTH
    # =====================================================

    mean_growth = (
        y_train.mean()
    )

    mean_growth_predictions = np.repeat(
        mean_growth,
        len(test)
    )

    mean_prices = reconstruct_prices(
        previous_prices,
        mean_growth_predictions
    )

    mean_mae, mean_rmse, mean_mape = (
        evaluate_predictions(
            actual_prices,
            mean_prices
        )
    )


    # =====================================================
    # MODEL 3: PREVIOUS YEAR GROWTH
    # =====================================================

    seasonal_growth = (
        test["Growth_Lag_4"].values
    )

    seasonal_prices = reconstruct_prices(
        previous_prices,
        seasonal_growth
    )

    seasonal_mae, seasonal_rmse, seasonal_mape = (
        evaluate_predictions(
            actual_prices,
            seasonal_prices
        )
    )


    # =====================================================
    # MODEL 4: HISTORICAL MEDIAN GROWTH
    # =====================================================

    median_growth = (
        y_train.median()
    )

    median_growth_predictions = np.repeat(
        median_growth,
        len(test)
    )

    median_prices = reconstruct_prices(
        previous_prices,
        median_growth_predictions
    )

    median_mae, median_rmse, median_mape = (
        evaluate_predictions(
            actual_prices,
            median_prices
        )
    )


    # =====================================================
    # SAVE FOLD RESULTS
    # =====================================================

    models = [
        (
            "Random Forest Growth",
            rf_mae,
            rf_rmse,
            rf_mape
        ),
        (
            "Historical Mean Growth",
            mean_mae,
            mean_rmse,
            mean_mape
        ),
        (
            "Seasonal Growth Lag 4",
            seasonal_mae,
            seasonal_rmse,
            seasonal_mape
        ),
        (
            "Historical Median Growth",
            median_mae,
            median_rmse,
            median_mape
        )
    ]


    for (
        model_name,
        mae,
        rmse,
        mape
    ) in models:

        fold_results.append({

            "Fold": fold,

            "Model": model_name,

            "MAE": mae,

            "RMSE": rmse,

            "MAPE": mape

        })


# =========================================================
# 7. CREATE RESULTS DATAFRAME
# =========================================================

fold_results_df = pd.DataFrame(
    fold_results
)


print("\n\n--- Fold-Level Results ---")

print(
    fold_results_df.to_string(
        index=False
    )
)


# =========================================================
# 8. MODEL SUMMARY
# =========================================================

comparison = (
    fold_results_df
    .groupby("Model")
    .agg(
        MAE=("MAE", "mean"),
        RMSE=("RMSE", "mean"),
        MAPE=("MAPE", "mean"),
        MAPE_Std=("MAPE", "std"),
        MAE_Std=("MAE", "std"),
        RMSE_Std=("RMSE", "std")
    )
    .reset_index()
)


comparison = comparison.sort_values(
    "MAPE"
).reset_index(drop=True)


print("\n--- Benchmark Model Comparison ---")

print(
    comparison.to_string(
        index=False
    )
)


# =========================================================
# 9. IDENTIFY BEST MODEL
# =========================================================

best_model = comparison.iloc[0]


print("\n--- Best Benchmark Model ---")

print(
    "Model:",
    best_model["Model"]
)

print(
    f"Mean MAE : {best_model['MAE']:.2f}"
)

print(
    f"Mean RMSE: {best_model['RMSE']:.2f}"
)

print(
    f"Mean MAPE: {best_model['MAPE']:.2f} %"
)


# =========================================================
# 10. SAVE RESULTS
# =========================================================

fold_file = (
    OUTPUT_DIR
    / "benchmark_fold_results.csv"
)

comparison_file = (
    OUTPUT_DIR
    / "benchmark_model_comparison.csv"
)


fold_results_df.to_csv(
    fold_file,
    index=False
)

comparison.to_csv(
    comparison_file,
    index=False
)


print("\n--- Benchmark Results Saved ---")

print(
    "Fold results:"
)

print(fold_file)

print(
    "\nModel comparison:"
)

print(comparison_file)


# =========================================================
# 11. RESEARCH INTERPRETATION
# =========================================================

rf_result = comparison[
    comparison["Model"]
    == "Random Forest Growth"
].iloc[0]


seasonal_result = comparison[
    comparison["Model"]
    == "Seasonal Growth Lag 4"
].iloc[0]


improvement = (
    (
        seasonal_result["MAPE"]
        - rf_result["MAPE"]
    )
    / seasonal_result["MAPE"]
) * 100


print("\n--- Research Interpretation ---")

print(
    f"Random Forest MAPE improvement "
    f"over Seasonal Growth benchmark: "
    f"{improvement:.2f}%"
)

print(
    "\nThe benchmark experiment evaluates "
    "whether the proposed machine-learning "
    "approach provides predictive value beyond "
    "simple historical and seasonal forecasting rules."
)