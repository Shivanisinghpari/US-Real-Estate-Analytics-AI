import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor


# =========================================================
# 1. Locate project files
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

input_file = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cleaned_us_housing_market.csv"
)


# =========================================================
# 2. Load historical housing data
# =========================================================

df = pd.read_csv(input_file)

df["Date"] = pd.to_datetime(df["Date"])

df = (
    df.sort_values("Date")
      .reset_index(drop=True)
)


print("\n--- Future Forecast: Data Loaded ---")
print("Dataset shape:", df.shape)

print(
    "Historical date range:",
    df["Date"].min(),
    "to",
    df["Date"].max()
)


# =========================================================
# 3. Create historical growth features
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
# 4. Define ML features
# =========================================================

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


# =========================================================
# 5. Prepare training dataset
# =========================================================

df_model = df[
    features + ["Target_Growth"]
].dropna().copy()


X_train = df_model[features]

y_train = df_model["Target_Growth"]


print("\n--- Training Future Forecast Model ---")
print("Training observations:", len(X_train))


# =========================================================
# 6. Train final Random Forest Growth Model
# =========================================================

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

print("Model training completed.")


# =========================================================
# 7. Prepare historical values for recursive forecasting
# =========================================================

history = df[
    ["Date", "Median_Price"]
].copy()

history = (
    history
    .sort_values("Date")
    .reset_index(drop=True)
)


last_date = history["Date"].iloc[-1]
last_price = history["Median_Price"].iloc[-1]

print("\n--- Forecast Starting Point ---")
print("Last historical date:", last_date)
print("Last historical price:", last_price)


# =========================================================
# 8. Forecast next 8 quarters
# =========================================================

forecast_horizon = 8

future_results = []


for step in range(1, forecast_horizon + 1):

    # -----------------------------------------------------
    # Create next quarter date
    # -----------------------------------------------------

    future_date = (
        last_date
        + pd.DateOffset(months=3)
    )

    future_year = future_date.year

    future_quarter = future_date.quarter


    # -----------------------------------------------------
    # Historical prices
    # -----------------------------------------------------

    prices = history["Median_Price"].tolist()


    # Previous-quarter price
    price_lag_1 = prices[-1]


    # Price four quarters earlier
    price_lag_4 = prices[-4]


    # -----------------------------------------------------
    # Growth features
    # -----------------------------------------------------

    growth_lag_1 = (
        (prices[-1] - prices[-2])
        / prices[-2]
        * 100
    )


    growth_lag_4 = (
        (prices[-1] - prices[-5])
        / prices[-5]
        * 100
    )


    # -----------------------------------------------------
    # Rolling four-quarter mean
    # -----------------------------------------------------

    rolling_mean_4 = np.mean(
        prices[-4:]
    )


    # -----------------------------------------------------
    # Time index
    # -----------------------------------------------------

    time_index = (
        len(history)
    )


    # -----------------------------------------------------
    # Create feature row
    # -----------------------------------------------------

    future_features = pd.DataFrame([{
        "Year": future_year,
        "Quarter": future_quarter,
        "Time_Index": time_index,
        "Price_Lag_1": price_lag_1,
        "Price_Lag_4": price_lag_4,
        "Growth_Lag_1": growth_lag_1,
        "Growth_Lag_4": growth_lag_4,
        "Rolling_Mean_4": rolling_mean_4
    }])


    # -----------------------------------------------------
    # Predict quarterly growth
    # -----------------------------------------------------

    predicted_growth = model.predict(
        future_features[features]
    )[0]


    # -----------------------------------------------------
    # Convert growth into predicted price
    # -----------------------------------------------------

    predicted_price = (
        price_lag_1
        * (1 + predicted_growth / 100)
    )


    # -----------------------------------------------------
    # Save forecast
    # -----------------------------------------------------

    future_results.append({
        "Date": future_date,
        "Year": future_year,
        "Quarter": future_quarter,
        "Predicted_Growth": predicted_growth,
        "Predicted_Price": predicted_price
    })


    # -----------------------------------------------------
    # Add prediction to history
    # -----------------------------------------------------

    history.loc[len(history)] = [
        future_date,
        predicted_price
    ]


    # Update last date
    last_date = future_date


# =========================================================
# 9. Convert results to DataFrame
# =========================================================

forecast_df = pd.DataFrame(
    future_results
)


print("\n--- Future Price Forecast ---")

print(
    forecast_df.to_string(
        index=False
    )
)


# =========================================================
# 10. Save forecast
# =========================================================

output_file = (
    PROJECT_ROOT
    / "data"
    / "forecasts"
    / "future_price_forecast.csv"
)

output_file.parent.mkdir(
    parents=True,
    exist_ok=True
)

forecast_df.to_csv(
    output_file,
    index=False
)


print("\n--- Future Forecast Saved ---")

print(
    "Results saved to:"
)

print(output_file)