import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


# ---------------------------------------------------------
# 1. Locate project
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

input_file = (
    PROJECT_ROOT
    / "data"
    / "forecasts"
    / "walk_forward_predictions.csv"
)

output_folder = (
    PROJECT_ROOT
    / "data"
    / "forecasts"
)

output_folder.mkdir(
    parents=True,
    exist_ok=True
)


# ---------------------------------------------------------
# 2. Load walk-forward predictions
# ---------------------------------------------------------

df = pd.read_csv(input_file)

print("\n--- Residual Analysis: Data Loaded ---")
print("Shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())


# ---------------------------------------------------------
# 3. Convert date
# ---------------------------------------------------------

if "Date" in df.columns:
    df["Date"] = pd.to_datetime(df["Date"])

else:
    raise ValueError(
        "Date column was not found in walk_forward_predictions.csv"
    )


# ---------------------------------------------------------
# 4. Identify actual and predicted price columns
# ---------------------------------------------------------

actual_candidates = [
    "Actual_Price",
    "Actual",
    "actual_price",
    "actual"
]

predicted_candidates = [
    "Predicted_Price",
    "Predicted",
    "predicted_price",
    "predicted"
]


actual_column = None
predicted_column = None


for column in actual_candidates:

    if column in df.columns:
        actual_column = column
        break


for column in predicted_candidates:

    if column in df.columns:
        predicted_column = column
        break


if actual_column is None:

    raise ValueError(
        "Could not find actual price column."
    )


if predicted_column is None:

    raise ValueError(
        "Could not find predicted price column."
    )


print("\nActual column:", actual_column)
print("Predicted column:", predicted_column)


# ---------------------------------------------------------
# 5. Calculate residuals
# ---------------------------------------------------------

df["Residual"] = (
    df[actual_column]
    - df[predicted_column]
)

df["Absolute_Error"] = (
    df["Residual"].abs()
)

df["Percentage_Error"] = (
    df["Residual"]
    / df[actual_column]
    * 100
)

df["Absolute_Percentage_Error"] = (
    df["Percentage_Error"].abs()
)


# ---------------------------------------------------------
# 6. Error direction
# ---------------------------------------------------------

df["Prediction_Direction"] = np.where(
    df["Residual"] > 0,
    "Underprediction",
    np.where(
        df["Residual"] < 0,
        "Overprediction",
        "Exact"
    )
)


# ---------------------------------------------------------
# 7. Display residual summary
# ---------------------------------------------------------

print("\n--- Residual Summary ---")

print(
    df[
        [
            "Date",
            actual_column,
            predicted_column,
            "Residual",
            "Absolute_Error",
            "Percentage_Error",
            "Prediction_Direction"
        ]
    ].head(10)
)


# ---------------------------------------------------------
# 8. Overall residual statistics
# ---------------------------------------------------------

mean_residual = df["Residual"].mean()

mean_absolute_error = (
    df["Absolute_Error"].mean()
)

rmse = np.sqrt(
    np.mean(
        df["Residual"] ** 2
    )
)

mean_percentage_error = (
    df["Percentage_Error"].mean()
)

mape = (
    df["Absolute_Percentage_Error"].mean()
)


print("\n--- Overall Error Statistics ---")

print(
    f"Mean Residual: {mean_residual:.2f}"
)

print(
    f"Mean Absolute Error: {mean_absolute_error:.2f}"
)

print(
    f"RMSE: {rmse:.2f}"
)

print(
    f"Mean Percentage Error: "
    f"{mean_percentage_error:.2f}%"
)

print(
    f"MAPE: {mape:.2f}%"
)


# ---------------------------------------------------------
# 9. Prediction bias
# ---------------------------------------------------------

underprediction_count = (
    df["Prediction_Direction"]
    .eq("Underprediction")
    .sum()
)

overprediction_count = (
    df["Prediction_Direction"]
    .eq("Overprediction")
    .sum()
)


print("\n--- Prediction Bias ---")

print(
    "Underpredictions:",
    underprediction_count
)

print(
    "Overpredictions:",
    overprediction_count
)

print(
    "Total predictions:",
    len(df)
)


# ---------------------------------------------------------
# 10. Largest prediction errors
# ---------------------------------------------------------

largest_errors = (
    df.sort_values(
        "Absolute_Error",
        ascending=False
    )
    .head(10)
)


print("\n--- Largest Prediction Errors ---")

print(
    largest_errors[
        [
            "Date",
            actual_column,
            predicted_column,
            "Residual",
            "Absolute_Error",
            "Absolute_Percentage_Error"
        ]
    ].to_string(index=False)
)


# ---------------------------------------------------------
# 11. Error by quarter
# ---------------------------------------------------------

df["Quarter"] = (
    df["Date"]
    .dt.quarter
)

quarter_summary = (
    df.groupby("Quarter")
    .agg(
        Mean_Absolute_Error=(
            "Absolute_Error",
            "mean"
        ),
        Mean_MAPE=(
            "Absolute_Percentage_Error",
            "mean"
        ),
        Mean_Residual=(
            "Residual",
            "mean"
        ),
        Observations=(
            "Residual",
            "count"
        )
    )
    .reset_index()
)


print("\n--- Error by Quarter ---")

print(
    quarter_summary.to_string(
        index=False
    )
)


# ---------------------------------------------------------
# 12. Error by prediction direction
# ---------------------------------------------------------

direction_summary = (
    df["Prediction_Direction"]
    .value_counts()
    .reset_index()
)

direction_summary.columns = [
    "Prediction_Direction",
    "Count"
]


print("\n--- Prediction Direction ---")

print(
    direction_summary.to_string(
        index=False
    )
)


# ---------------------------------------------------------
# 13. Save detailed residual dataset
# ---------------------------------------------------------

residual_file = (
    output_folder
    / "residual_analysis.csv"
)

df.to_csv(
    residual_file,
    index=False
)


# ---------------------------------------------------------
# 14. Save residual summary
# ---------------------------------------------------------

summary = pd.DataFrame({
    "Metric": [
        "Mean Residual",
        "Mean Absolute Error",
        "RMSE",
        "Mean Percentage Error",
        "MAPE",
        "Underpredictions",
        "Overpredictions",
        "Total Predictions"
    ],
    "Value": [
        mean_residual,
        mean_absolute_error,
        rmse,
        mean_percentage_error,
        mape,
        underprediction_count,
        overprediction_count,
        len(df)
    ]
})


summary_file = (
    output_folder
    / "residual_summary.csv"
)

summary.to_csv(
    summary_file,
    index=False
)


# ---------------------------------------------------------
# 15. Save quarter-level analysis
# ---------------------------------------------------------

quarter_file = (
    output_folder
    / "error_by_quarter.csv"
)

quarter_summary.to_csv(
    quarter_file,
    index=False
)


# ---------------------------------------------------------
# 16. Residual visualization
# ---------------------------------------------------------

plt.figure(
    figsize=(12, 6)
)

plt.axhline(
    0,
    linestyle="--"
)

plt.scatter(
    df["Date"],
    df["Residual"],
    alpha=0.7
)

plt.xlabel("Date")
plt.ylabel("Prediction Residual")

plt.title(
    "Random Forest Growth Model: Residual Analysis"
)

plt.grid(True)

plt.tight_layout()


plot_file = (
    output_folder
    / "residual_analysis.png"
)

plt.savefig(
    plot_file,
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ---------------------------------------------------------
# 17. Final output
# ---------------------------------------------------------

print("\n--- Residual Analysis Saved ---")

print(
    "Detailed residual data:"
)

print(residual_file)

print(
    "\nResidual summary:"
)

print(summary_file)

print(
    "\nQuarter analysis:"
)

print(quarter_file)

print(
    "\nResidual visualization:"
)

print(plot_file)


# ---------------------------------------------------------
# 18. Research interpretation
# ---------------------------------------------------------

print(
    "\n--- Research Interpretation ---"
)

if mean_residual > 0:

    print(
        "The model shows an overall tendency "
        "to underestimate actual prices."
    )

elif mean_residual < 0:

    print(
        "The model shows an overall tendency "
        "to overestimate actual prices."
    )

else:

    print(
        "The model shows no overall residual bias."
    )

print(
    "Residual analysis examines whether forecasting "
    "errors are systematic across time and seasons."
)