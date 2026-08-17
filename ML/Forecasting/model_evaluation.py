import pandas as pd


# ---------------------------------------------------------
# Model Evaluation Results
# ---------------------------------------------------------

results = pd.DataFrame({
    "Model": [
        "ARIMA(1,1,1)",
        "SARIMA(1,1,1)(1,1,1,4)"
    ],
    "MAE": [
        59836.50,
        36308.37
    ],
    "RMSE": [
        71855.06,
        47377.76
    ],
    "MAPE": [
        15.55,
        9.26
    ]
})


print("--- Forecasting Model Comparison ---")
print(results.to_string(index=False))


# ---------------------------------------------------------
# Identify the best model based on MAPE
# ---------------------------------------------------------

best_model = results.loc[
    results["MAPE"].idxmin(),
    "Model"
]

best_mape = results["MAPE"].min()

print("\n--- Selected Model ---")
print("Best model based on MAPE:", best_model)
print(f"MAPE: {best_mape:.2f} %")


# ---------------------------------------------------------
# Calculate relative MAPE improvement
# ---------------------------------------------------------

arima_mape = results.loc[
    results["Model"] == "ARIMA(1,1,1)",
    "MAPE"
].iloc[0]

sarima_mape = results.loc[
    results["Model"] == "SARIMA(1,1,1)(1,1,1,4)",
    "MAPE"
].iloc[0]

improvement = (
    (arima_mape - sarima_mape)
    / arima_mape
) * 100

print("\n--- Model Improvement ---")
print(f"MAPE improvement from ARIMA to SARIMA: {improvement:.2f} %")