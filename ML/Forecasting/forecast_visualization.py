import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------
# Load Historical Data
# ---------------------------------------------------------

historical_path = "data/processed/cleaned_us_housing_market.csv"

historical_df = pd.read_csv(historical_path)

historical_df["Date"] = pd.to_datetime(historical_df["Date"])


# ---------------------------------------------------------
# Load SARIMA Forecast
# ---------------------------------------------------------

forecast_path = "data/forecasts/sarima_8_quarter_forecast.csv"

forecast_df = pd.read_csv(forecast_path)

forecast_df["Date"] = pd.to_datetime(forecast_df["Date"])


# ---------------------------------------------------------
# Create Forecast Visualization
# ---------------------------------------------------------

plt.figure(figsize=(14, 7))

# Historical median prices
plt.plot(
    historical_df["Date"],
    historical_df["Median_Price"],
    label="Historical Median Price"
)

# SARIMA forecast
plt.plot(
    forecast_df["Date"],
    forecast_df["Forecast"],
    linestyle="--",
    label="SARIMA Forecast"
)

# Confidence interval
plt.fill_between(
    forecast_df["Date"],
    forecast_df["Lower_CI"],
    forecast_df["Upper_CI"],
    alpha=0.2,
    label="95% Confidence Interval"
)


# ---------------------------------------------------------
# Formatting
# ---------------------------------------------------------

# ---------------------------------------------------------
# Forecast Boundary
# ---------------------------------------------------------

forecast_start = forecast_df["Date"].iloc[0]

plt.axvline(
    forecast_start,
    linestyle=":",
    linewidth=2,
    label="Forecast Start"
)


# ---------------------------------------------------------
# Formatting
# ---------------------------------------------------------

plt.title(
    "U.S. Median House Price: Historical Trend and 8-Quarter SARIMA Forecast"
)

plt.xlabel("Date")
plt.ylabel("Median House Price ($)")

plt.legend()

plt.grid(True, alpha=0.3)

plt.tight_layout()

plt.show()