import pandas as pd
from pathlib import Path


# ---------------------------------------------------------
# 1. Load cleaned housing market data
# ---------------------------------------------------------

project_root = Path(__file__).resolve().parents[2]

file_path = (
    project_root
    / "data"
    / "processed"
    / "cleaned_us_housing_market.csv"
)

df = pd.read_csv(file_path)

df["Date"] = pd.to_datetime(df["Date"])

df = df.sort_values("Date")

print("--- ARIMA Baseline: Data Loaded ---")
print("Dataset shape:", df.shape)
print("Date range:", df["Date"].min(), "to", df["Date"].max())
print("Columns:", df.columns.tolist())

# ---------------------------------------------------------
# 2. Train / Test Split
# ---------------------------------------------------------

# Use Median_Price as the forecasting target
series = df.set_index("Date")["Median_Price"]

# Keep the final 51 observations for testing
test_size = 51

train = series.iloc[:-test_size]
test = series.iloc[-test_size:]

print("\n--- Train/Test Split ---")
print("Training observations:", len(train))
print("Testing observations:", len(test))

print("Training period:")
print(train.index.min(), "to", train.index.max())

print("Testing period:")
print(test.index.min(), "to", test.index.max())

# ---------------------------------------------------------
# 3. ARIMA Baseline Model
# ---------------------------------------------------------

from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np


print("\n--- Training ARIMA(1,1,1) Baseline ---")

# Create and fit the ARIMA model
model = ARIMA(
    train,
    order=(1, 1, 1)
)

model_fit = model.fit()

print(model_fit.summary())


# ---------------------------------------------------------
# 4. Forecast the Test Period
# ---------------------------------------------------------

forecast = model_fit.forecast(steps=len(test))

# Make sure the forecast uses the same dates as the test data
forecast.index = test.index


print("\n--- ARIMA Forecast Generated ---")
print(forecast.head())
print("...")
print(forecast.tail())


# ---------------------------------------------------------
# 5. Evaluate the ARIMA Baseline
# ---------------------------------------------------------

mae = mean_absolute_error(test, forecast)

rmse = np.sqrt(
    mean_squared_error(test, forecast)
)

mape = np.mean(
    np.abs((test - forecast) / test)
) * 100


print("\n--- ARIMA Baseline Evaluation ---")
print(f"MAE : {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAPE: {mape:.2f} %")

# ---------------------------------------------------------
# 6. Save ARIMA Evaluation Results
# ---------------------------------------------------------

results = pd.DataFrame({
    "Model": ["ARIMA(1,1,1)"],
    "MAE": [mae],
    "RMSE": [rmse],
    "MAPE": [mape]
})

results_file = (
    project_root
    / "data"
    / "forecasts"
    / "arima_baseline_results.csv"
)

results.to_csv(
    results_file,
    index=False
)

print("\n--- ARIMA Results Saved ---")
print(results.to_string(index=False))

print("\nResults saved to:")
print(results_file)