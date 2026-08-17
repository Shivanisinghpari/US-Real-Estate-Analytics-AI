import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error


# -----------------------------------
# 1. LOAD CLEANED DATA
# -----------------------------------

file_path = "data/processed/cleaned_us_housing_market.csv"

df = pd.read_csv(file_path)

df["Date"] = pd.to_datetime(df["Date"])

df = df.sort_values("Date")

df = df.set_index("Date")

# Quarterly data: January, April, July, October
df = df.asfreq("QS")

price_series = df["Median_Price"]


# -----------------------------------
# 2. TRAIN / TEST SPLIT
# -----------------------------------

train_size = int(len(price_series) * 0.80)

train = price_series.iloc[:train_size]
test = price_series.iloc[train_size:]

print("\n--- Train/Test Split ---")

print("Training observations:", len(train))
print("Testing observations:", len(test))

print("Training period:")
print(train.index.min(), "to", train.index.max())

print("Testing period:")
print(test.index.min(), "to", test.index.max())


# -----------------------------------
# 3. SARIMA MODEL FOR EVALUATION
# -----------------------------------

model = SARIMAX(
    train,
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 4),
    enforce_stationarity=False,
    enforce_invertibility=False
)

model_fit = model.fit(disp=False)


# -----------------------------------
# 4. FORECAST TEST PERIOD
# -----------------------------------

test_forecast = model_fit.forecast(
    steps=len(test)
)

test_forecast.index = test.index


# -----------------------------------
# 5. EVALUATE SARIMA
# -----------------------------------

mae = mean_absolute_error(
    test,
    test_forecast
)

rmse = np.sqrt(
    mean_squared_error(
        test,
        test_forecast
    )
)

mape = np.mean(
    np.abs(
        (test - test_forecast) / test
    )
) * 100


print("\n--- SARIMA Model Evaluation ---")

print("MAE :", round(mae, 2))

print("RMSE:", round(rmse, 2))

print("MAPE:", round(mape, 2), "%")


# -----------------------------------
# 6. TRAIN FINAL MODEL ON ALL DATA
# -----------------------------------

print("\n--- Training Final SARIMA Model ---")

final_model = SARIMAX(
    price_series,
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 4),
    enforce_stationarity=False,
    enforce_invertibility=False
)

final_model_fit = final_model.fit(disp=False)


# -----------------------------------
# 7. FORECAST NEXT 8 QUARTERS
# -----------------------------------

forecast_steps = 8

future_forecast = final_model_fit.get_forecast(
    steps=forecast_steps
)

forecast_mean = future_forecast.predicted_mean

confidence_interval = future_forecast.conf_int()


# -----------------------------------
# 8. CREATE FORECAST DATAFRAME
# -----------------------------------

forecast_df = pd.DataFrame({
    "Date": forecast_mean.index,
    "Forecast": forecast_mean.values,
    "Lower_CI": confidence_interval.iloc[:, 0].values,
    "Upper_CI": confidence_interval.iloc[:, 1].values
})

forecast_df["Date"] = pd.to_datetime(
    forecast_df["Date"]
)


print("\n--- Forecast for Next 8 Quarters ---")

print(
    forecast_df.to_string(index=False)
)


# -----------------------------------
# 9. SAVE FORECAST
# -----------------------------------

forecast_directory = "data/forecasts"

os.makedirs(
    forecast_directory,
    exist_ok=True
)

forecast_file = (
    "data/forecasts/"
    "sarima_8_quarter_forecast.csv"
)

forecast_df.to_csv(
    forecast_file,
    index=False
)

print(
    "\nForecast saved to:",
    forecast_file
)


# -----------------------------------
# 10. FINAL FORECAST GRAPH
# -----------------------------------

plt.figure(figsize=(14, 7))

plt.plot(
    price_series.index,
    price_series,
    label="Historical Median Price"
)

plt.plot(
    forecast_df["Date"],
    forecast_df["Forecast"],
    linestyle="--",
    label="SARIMA Forecast"
)

plt.fill_between(
    forecast_df["Date"],
    forecast_df["Lower_CI"],
    forecast_df["Upper_CI"],
    alpha=0.2,
    label="95% Confidence Interval"
)

plt.axvline(
    price_series.index[-1],
    linestyle=":",
    label="Forecast Start"
)

plt.title(
    "U.S. Median House Price: Historical Data and SARIMA Forecast"
)

plt.xlabel("Year")

plt.ylabel(
    "Median House Price ($)"
)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.show()     