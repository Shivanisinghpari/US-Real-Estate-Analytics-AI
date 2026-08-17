import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor


# =========================================================
# 1. Locate project files
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cleaned_us_housing_market.csv"
)

FORECAST_FILE = (
    PROJECT_ROOT
    / "data"
    / "forecasts"
    / "future_price_forecast.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "forecasts"
    / "future_forecast_with_uncertainty.csv"
)


# =========================================================
# 2. Load historical data
# =========================================================

df = pd.read_csv(INPUT_FILE)

df["Date"] = pd.to_datetime(df["Date"])

df = (
    df
    .sort_values("Date")
    .reset_index(drop=True)
)

print("\n--- Historical Data Loaded ---")
print("Observations:", len(df))
print(
    "Historical period:",
    df["Date"].min(),
    "to",
    df["Date"].max()
)


# =========================================================
# 3. Create forecasting features
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

df["Target_Growth"] = (
    df["Median_Price"]
    .pct_change()
    * 100
)


# =========================================================
# 4. Define model features
# =========================================================

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


# =========================================================
# 5. Prepare modelling data
# =========================================================

model_df = (
    df[
        FEATURES
        + ["Target_Growth", "Median_Price", "Date"]
    ]
    .dropna()
    .reset_index(drop=True)
)


# =========================================================
# 6. Generate walk-forward residuals
# =========================================================
#
# We deliberately do NOT calculate uncertainty from the
# training predictions.
#
# Instead, every prediction is generated using only
# observations that occurred before that prediction date.
#
# This produces out-of-sample historical errors.
# =========================================================

MIN_TRAIN_SIZE = 100

residuals = []

print("\n--- Generating Walk-Forward Residuals ---")

for i in range(MIN_TRAIN_SIZE, len(model_df)):

    train = model_df.iloc[:i]
    test = model_df.iloc[i:i + 1]

    X_train = train[FEATURES]
    y_train = train["Target_Growth"]

    X_test = test[FEATURES]
    actual_growth = test["Target_Growth"].iloc[0]

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

    predicted_growth = model.predict(
        X_test
    )[0]

    residual = (
        actual_growth
        - predicted_growth
    )

    residuals.append(residual)


residuals = np.array(residuals)

print(
    "Walk-forward predictions:",
    len(residuals)
)

print(
    "Residual mean:",
    f"{np.mean(residuals):.4f}%"
)

print(
    "Residual standard deviation:",
    f"{np.std(residuals):.4f}%"
)


# =========================================================
# 7. Calculate empirical residual quantiles
# =========================================================

lower_residual = np.percentile(
    residuals,
    2.5
)

upper_residual = np.percentile(
    residuals,
    97.5
)

print("\n--- Empirical Residual Range ---")

print(
    "2.5th percentile:",
    f"{lower_residual:.4f}%"
)

print(
    "97.5th percentile:",
    f"{upper_residual:.4f}%"
)


# =========================================================
# 8. Load future point forecasts
# =========================================================

forecast_df = pd.read_csv(
    FORECAST_FILE
)

forecast_df["Date"] = pd.to_datetime(
    forecast_df["Date"]
)

print("\n--- Future Forecast Loaded ---")

print(
    "Forecast observations:",
    len(forecast_df)
)


# =========================================================
# 9. Build uncertainty intervals
# =========================================================
#
# The point forecast remains unchanged.
#
# We apply the historical out-of-sample growth-error
# distribution to each future forecast.
#
# This gives an empirical 95% prediction interval.
# =========================================================

forecast_df[
    "Predicted_Growth_Lower_95"
] = (
    forecast_df["Predicted_Growth"]
    + lower_residual
)

forecast_df[
    "Predicted_Growth_Upper_95"
] = (
    forecast_df["Predicted_Growth"]
    + upper_residual
)


# =========================================================
# 10. Convert growth intervals into price intervals
# =========================================================

previous_price = None

for i in range(len(forecast_df)):

    if i == 0:

        previous_price = (
            df["Median_Price"].iloc[-1]
        )

    else:

        previous_price = (
            forecast_df[
                "Predicted_Price"
            ].iloc[i - 1]
        )

    forecast_df.loc[
        i,
        "Predicted_Price_Lower_95"
    ] = (
        previous_price
        * (
            1
            + forecast_df[
                "Predicted_Growth_Lower_95"
            ].iloc[i]
            / 100
        )
    )

    forecast_df.loc[
        i,
        "Predicted_Price_Upper_95"
    ] = (
        previous_price
        * (
            1
            + forecast_df[
                "Predicted_Growth_Upper_95"
            ].iloc[i]
            / 100
        )
    )


# =========================================================
# 11. Display final uncertainty results
# =========================================================

print("\n--- Future Forecast With 95% Prediction Interval ---")

display_columns = [
    "Date",
    "Predicted_Price",
    "Predicted_Price_Lower_95",
    "Predicted_Price_Upper_95"
]

print(
    forecast_df[
        display_columns
    ].to_string(
        index=False,
        formatters={
            "Predicted_Price":
                "${:,.2f}".format,

            "Predicted_Price_Lower_95":
                "${:,.2f}".format,

            "Predicted_Price_Upper_95":
                "${:,.2f}".format
        }
    )
)


# =========================================================
# 12. Save final uncertainty dataset
# =========================================================

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

forecast_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n--- Forecast Uncertainty Saved ---")

print(
    "Results saved to:"
)

print(OUTPUT_FILE)