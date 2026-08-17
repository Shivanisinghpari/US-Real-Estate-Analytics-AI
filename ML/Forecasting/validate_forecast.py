import pandas as pd


# ---------------------------------------------------------
# Load Forecast
# ---------------------------------------------------------

file_path = "data/forecasts/sarima_8_quarter_forecast.csv"

forecast_df = pd.read_csv(file_path)

forecast_df["Date"] = pd.to_datetime(forecast_df["Date"])


# ---------------------------------------------------------
# Basic Validation
# ---------------------------------------------------------

print("--- Forecast Validation ---")

print("Number of forecast periods:", len(forecast_df))

print("\nDate range:")
print(
    forecast_df["Date"].min(),
    "to",
    forecast_df["Date"].max()
)

print("\nMissing values:")
print(forecast_df.isnull().sum())


# ---------------------------------------------------------
# Confidence Interval Validation
# ---------------------------------------------------------

interval_check = (
    (forecast_df["Lower_CI"] <= forecast_df["Forecast"])
    &
    (forecast_df["Forecast"] <= forecast_df["Upper_CI"])
)

print("\nConfidence interval check:")

if interval_check.all():
    print("PASS: All forecasts are inside their confidence intervals.")
else:
    print("FAIL: Some forecasts fall outside their confidence intervals.")


# ---------------------------------------------------------
# Date Sequence Validation
# ---------------------------------------------------------

date_differences = forecast_df["Date"].diff().dropna()

print("\nDate differences:")
print(date_differences.value_counts())

if len(forecast_df) == 8:
    print("\nPASS: Exactly 8 forecast periods found.")
else:
    print("\nWARNING: Expected 8 forecast periods.")


# ---------------------------------------------------------
# Final Forecast Table
# ---------------------------------------------------------

print("\n--- Validated Forecast ---")
print(forecast_df.to_string(index=False))