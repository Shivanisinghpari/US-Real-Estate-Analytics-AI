import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# =========================================================
# 1. Locate project files
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

historical_file = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cleaned_us_housing_market.csv"
)

forecast_file = (
    PROJECT_ROOT
    / "data"
    / "forecasts"
    / "future_price_forecast.csv"
)


# =========================================================
# 2. Load historical data
# =========================================================

historical = pd.read_csv(
    historical_file
)

historical["Date"] = pd.to_datetime(
    historical["Date"]
)

historical = (
    historical
    .sort_values("Date")
    .reset_index(drop=True)
)


print("\n--- Historical Data Loaded ---")

print(
    "Historical observations:",
    len(historical)
)

print(
    "Historical period:",
    historical["Date"].min(),
    "to",
    historical["Date"].max()
)


# =========================================================
# 3. Load future forecast
# =========================================================

forecast = pd.read_csv(
    forecast_file
)

forecast["Date"] = pd.to_datetime(
    forecast["Date"]
)

forecast = (
    forecast
    .sort_values("Date")
    .reset_index(drop=True)
)


print("\n--- Future Forecast Loaded ---")

print(
    "Forecast observations:",
    len(forecast)
)

print(
    "Forecast period:",
    forecast["Date"].min(),
    "to",
    forecast["Date"].max()
)


# =========================================================
# 4. Forecast starting point
# =========================================================

forecast_start = forecast["Date"].min()

last_actual_date = historical["Date"].max()

last_actual_price = historical[
    "Median_Price"
].iloc[-1]


print("\n--- Forecast Starting Point ---")

print(
    "Last actual date:",
    last_actual_date
)

print(
    "Last actual price:",
    f"${last_actual_price:,.2f}"
)

print(
    "Forecast starts:",
    forecast_start
)


# =========================================================
# 5. Create visualization
# =========================================================

plt.figure(
    figsize=(14, 7)
)


# ---------------------------------------------------------
# Historical prices
# ---------------------------------------------------------

plt.plot(
    historical["Date"],
    historical["Median_Price"],
    label="Historical Median Price",
    linewidth=2
)


# ---------------------------------------------------------
# Future forecast
# ---------------------------------------------------------

plt.plot(
    forecast["Date"],
    forecast["Predicted_Price"],
    label="Random Forest Growth Forecast",
    linewidth=2,
    linestyle="--"
)


# ---------------------------------------------------------
# Connect final actual price to first forecast
# ---------------------------------------------------------

plt.plot(
    [
        last_actual_date,
        forecast["Date"].iloc[0]
    ],
    [
        last_actual_price,
        forecast["Predicted_Price"].iloc[0]
    ],
    linestyle="--",
    linewidth=2
)


# =========================================================
# 6. Forecast start marker
# =========================================================

plt.axvline(
    forecast_start,
    linestyle=":",
    linewidth=2,
    label="Forecast Start"
)


# =========================================================
# 7. Highlight forecast region
# =========================================================

plt.axvspan(
    forecast_start,
    forecast["Date"].max(),
    alpha=0.08
)


# =========================================================
# 8. Labels
# =========================================================

plt.xlabel(
    "Date",
    fontsize=12
)

plt.ylabel(
    "Median House Price ($)",
    fontsize=12
)

plt.title(
    "U.S. Median House Price: Historical Data and 8-Quarter ML Forecast",
    fontsize=15
)


# =========================================================
# 9. Formatting
# =========================================================

plt.grid(
    True,
    alpha=0.3
)

plt.legend(
    loc="upper left"
)

plt.tight_layout()


# =========================================================
# 10. Save visualization
# =========================================================

output_file = (
    PROJECT_ROOT
    / "data"
    / "forecasts"
    / "future_forecast_visualization.png"
)

plt.savefig(
    output_file,
    dpi=300,
    bbox_inches="tight"
)


print("\n--- Visualization Saved ---")

print(
    "Visualization saved to:"
)

print(output_file)


# =========================================================
# 11. Display visualization
# =========================================================

plt.show()