import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)


# =========================================================
# 1. Locate project
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

input_file = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cleaned_us_housing_market.csv"
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


print("\n--- Ablation Study: Data Loaded ---")

print("Dataset shape:", df.shape)

print(
    "Date range:",
    df["Date"].min(),
    "to",
    df["Date"].max()
)


# =========================================================
# 3. Create features
# =========================================================

df["Time_Index"] = range(len(df))


# Previous quarter price
df["Price_Lag_1"] = (
    df["Median_Price"].shift(1)
)


# Same quarter previous year
df["Price_Lag_4"] = (
    df["Median_Price"].shift(4)
)


# Previous quarter growth
df["Growth_Lag_1"] = (
    df["Median_Price"]
    .pct_change()
    .shift(1)
    * 100
)


# Previous year growth
df["Growth_Lag_4"] = (
    df["Median_Price"]
    .pct_change(4)
    .shift(1)
    * 100
)


# Historical four-quarter rolling mean
df["Rolling_Mean_4"] = (
    df["Median_Price"]
    .shift(1)
    .rolling(4)
    .mean()
)


# Target
df["Target_Growth"] = (
    df["Median_Price"]
    .pct_change()
    * 100
)


# =========================================================
# 4. Define feature groups
# =========================================================

feature_groups = {

    "Price Features": [
        "Price_Lag_1",
        "Price_Lag_4"
    ],

    "Growth Features": [
        "Growth_Lag_1",
        "Growth_Lag_4"
    ],

    "Time Features": [
        "Year",
        "Time_Index"
    ],

    "Seasonal Features": [
        "Quarter"
    ],

    "Rolling Features": [
        "Rolling_Mean_4"
    ],

    "Price + Growth": [
        "Price_Lag_1",
        "Price_Lag_4",
        "Growth_Lag_1",
        "Growth_Lag_4"
    ],

    "Price + Growth + Time": [
        "Price_Lag_1",
        "Price_Lag_4",
        "Growth_Lag_1",
        "Growth_Lag_4",
        "Year",
        "Time_Index"
    ],

    "All Features": [
        "Year",
        "Quarter",
        "Time_Index",
        "Price_Lag_1",
        "Price_Lag_4",
        "Growth_Lag_1",
        "Growth_Lag_4",
        "Rolling_Mean_4"
    ]
}


# =========================================================
# 5. Prepare modeling dataset
# =========================================================

required_columns = [
    "Date",
    "Median_Price",
    "Target_Growth"
]

for group_features in feature_groups.values():
    required_columns.extend(group_features)

required_columns = list(dict.fromkeys(required_columns))

df_model = (
    df[required_columns]
    .dropna()
    .reset_index(drop=True)
)


print("\n--- Ablation Dataset ---")
print("Shape:", df_model.shape)


# =========================================================
# 6. Walk-forward validation settings
# =========================================================

N_FOLDS = 5
TEST_SIZE = 20


print("\n--- Ablation Walk-Forward Validation ---")

print("Folds:", N_FOLDS)

print("Test observations per fold:", TEST_SIZE)


# =========================================================
# 7. Store results
# =========================================================

fold_results = []


# =========================================================
# 8. Run ablation study
# =========================================================

for feature_group_name, features in feature_groups.items():

    print("\n" + "=" * 60)

    print(
        "Feature Configuration:",
        feature_group_name
    )

    print(
        "Features:",
        features
    )

    print("=" * 60)


    group_mae = []
    group_rmse = []
    group_mape = []


    # -----------------------------------------------------
    # Walk-forward folds
    # -----------------------------------------------------

    for fold in range(N_FOLDS):

        train_end = (
            len(df_model)
            - (N_FOLDS - fold) * TEST_SIZE
        )

        test_start = train_end

        test_end = test_start + TEST_SIZE


        train = df_model.iloc[:train_end]

        test = df_model.iloc[
            test_start:test_end
        ]


        X_train = train[features]

        X_test = test[features]

        y_train = train["Target_Growth"]

        y_test = test["Target_Growth"]


        # -------------------------------------------------
        # Train model
        # -------------------------------------------------

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


        # -------------------------------------------------
        # Predict growth
        # -------------------------------------------------

        predicted_growth = model.predict(
            X_test
        )


        # -------------------------------------------------
        # Reconstruct prices
        # -------------------------------------------------

        previous_prices = (
            test["Price_Lag_1"].values
        )

        predicted_prices = (
            previous_prices
            * (1 + predicted_growth / 100)
        )


        actual_prices = (
            test["Median_Price"].values
        )


        # -------------------------------------------------
        # Evaluation
        # -------------------------------------------------

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


        group_mae.append(mae)

        group_rmse.append(rmse)

        group_mape.append(mape)


        fold_results.append({

            "Feature_Set": feature_group_name,

            "Fold": fold + 1,

            "Training_End":
                train["Date"].max(),

            "Testing_Start":
                test["Date"].min(),

            "Testing_End":
                test["Date"].max(),

            "MAE": mae,

            "RMSE": rmse,

            "MAPE": mape

        })


        print(
            f"Fold {fold + 1}: "
            f"MAE={mae:.2f}, "
            f"RMSE={rmse:.2f}, "
            f"MAPE={mape:.2f}%"
        )


    # -----------------------------------------------------
    # Average performance
    # -----------------------------------------------------

    print("\nAverage Performance:")

    print(
        f"MAE : {np.mean(group_mae):.2f}"
    )

    print(
        f"RMSE: {np.mean(group_rmse):.2f}"
    )

    print(
        f"MAPE: {np.mean(group_mape):.2f}%"
    )


# =========================================================
# 9. Convert fold results to DataFrame
# =========================================================

fold_results_df = pd.DataFrame(
    fold_results
)


# =========================================================
# 10. Model comparison
# =========================================================

comparison = (
    fold_results_df
    .groupby("Feature_Set")
    .agg(
        MAE=("MAE", "mean"),
        RMSE=("RMSE", "mean"),
        MAPE=("MAPE", "mean"),
        MAPE_Std=("MAPE", "std")
    )
    .reset_index()
)


comparison = (
    comparison
    .sort_values("MAPE")
    .reset_index(drop=True)
)


print("\n\n" + "=" * 70)

print("--- ABLATION STUDY RESULTS ---")

print("=" * 70)

print(
    comparison.to_string(
        index=False
    )
)


# =========================================================
# 11. Identify best feature configuration
# =========================================================

best_row = comparison.iloc[0]


print("\n--- Best Feature Configuration ---")

print(
    "Feature Set:",
    best_row["Feature_Set"]
)

print(
    f"Mean MAE: {best_row['MAE']:.2f}"
)

print(
    f"Mean RMSE: {best_row['RMSE']:.2f}"
)

print(
    f"Mean MAPE: {best_row['MAPE']:.2f}%"
)

print(
    f"MAPE Std: {best_row['MAPE_Std']:.2f}"
)


# =========================================================
# 12. Save results
# =========================================================

output_folder = (
    PROJECT_ROOT
    / "data"
    / "forecasts"
)

output_folder.mkdir(
    parents=True,
    exist_ok=True
)


fold_file = (
    output_folder
    / "ablation_fold_results.csv"
)


comparison_file = (
    output_folder
    / "ablation_model_comparison.csv"
)


fold_results_df.to_csv(
    fold_file,
    index=False
)


comparison.to_csv(
    comparison_file,
    index=False
)


print("\n--- Ablation Study Saved ---")

print(
    "Fold results:"
)

print(fold_file)


print(
    "\nFeature comparison:"
)

print(comparison_file)


print(
    "\nResearch note:"
)

print(
    "The ablation study evaluates how different "
    "feature groups contribute to forecasting "
    "performance using chronological walk-forward "
    "validation."
)