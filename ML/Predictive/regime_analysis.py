import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


# =========================================================
# 1. Locate project
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

input_file = (
    PROJECT_ROOT
    / "data"
    / "forecasts"
    / "walk_forward_predictions.csv"
)

output_dir = (
    PROJECT_ROOT
    / "data"
    / "forecasts"
    / "regime_analysis"
)

output_dir.mkdir(parents=True, exist_ok=True)


# =========================================================
# 2. Load walk-forward predictions
# =========================================================

df = pd.read_csv(input_file)

df["Date"] = pd.to_datetime(df["Date"])

print("\n--- Regime Analysis: Data Loaded ---")
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())


# =========================================================
# 3. Keep Random Forest model
# =========================================================

df = df[
    df["Model"] == "Random Forest Growth"
].copy()

df = df.sort_values("Date")


print("\n--- Random Forest Predictions ---")
print("Observations:", len(df))


# =========================================================
# 4. Calculate prediction errors
# =========================================================

df["Residual"] = (
    df["Actual_Price"]
    - df["Predicted_Price"]
)

df["Absolute_Error"] = (
    df["Residual"].abs()
)

df["Percentage_Error"] = (
    df["Residual"]
    / df["Actual_Price"]
    * 100
)

df["Absolute_Percentage_Error"] = (
    df["Percentage_Error"].abs()
)


# =========================================================
# 5. Define market regimes
# =========================================================

def classify_regime(growth):

    if growth < 0:
        return "Declining"

    elif growth < 2:
        return "Stable"

    elif growth < 5:
        return "Moderate Growth"

    else:
        return "Rapid Growth"


df["Market_Regime"] = (
    df["Actual_Growth"]
    .apply(classify_regime)
)


# =========================================================
# 6. Display regime distribution
# =========================================================

print("\n--- Market Regime Distribution ---")

print(
    df["Market_Regime"]
    .value_counts()
)


# =========================================================
# 7. Calculate performance by regime
# =========================================================

regime_results = (
    df.groupby("Market_Regime")
    .agg(
        Observations=("Actual_Price", "count"),
        MAE=("Absolute_Error", "mean"),
        RMSE=(
            "Residual",
            lambda x: np.sqrt(np.mean(x ** 2))
        ),
        MAPE=(
            "Absolute_Percentage_Error",
            "mean"
        ),
        Mean_Residual=("Residual", "mean")
    )
    .reset_index()
)


# =========================================================
# 8. Order regimes logically
# =========================================================

regime_order = [
    "Declining",
    "Stable",
    "Moderate Growth",
    "Rapid Growth"
]

regime_results["Market_Regime"] = pd.Categorical(
    regime_results["Market_Regime"],
    categories=regime_order,
    ordered=True
)

regime_results = (
    regime_results
    .sort_values("Market_Regime")
    .reset_index(drop=True)
)


# =========================================================
# 9. Print results
# =========================================================

print("\n--- Performance by Market Regime ---")

print(
    regime_results.to_string(index=False)
)


# =========================================================
# 10. Identify hardest regime
# =========================================================

hardest_regime = regime_results.loc[
    regime_results["MAPE"].idxmax(),
    "Market_Regime"
]

best_regime = regime_results.loc[
    regime_results["MAPE"].idxmin(),
    "Market_Regime"
]

print("\n--- Regime Analysis Summary ---")

print(
    "Highest-error regime:",
    hardest_regime
)

print(
    "Lowest-error regime:",
    best_regime
)


# =========================================================
# 11. Save regime results
# =========================================================

results_file = (
    output_dir
    / "regime_performance.csv"
)

regime_results.to_csv(
    results_file,
    index=False
)

print("\nRegime performance saved to:")
print(results_file)


# =========================================================
# 12. Save detailed predictions
# =========================================================

detail_file = (
    output_dir
    / "regime_predictions.csv"
)

df.to_csv(
    detail_file,
    index=False
)

print("\nRegime predictions saved to:")
print(detail_file)


# =========================================================
# 13. Visualization
# =========================================================

plt.figure(figsize=(10, 6))

plt.bar(
    regime_results["Market_Regime"].astype(str),
    regime_results["MAPE"]
)

plt.xlabel("Market Regime")
plt.ylabel("MAPE (%)")

plt.title(
    "Random Forest Forecast Error Across Market Regimes"
)

plt.xticks(rotation=20)

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()


plot_file = (
    output_dir
    / "regime_mape_comparison.png"
)

plt.savefig(
    plot_file,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nVisualization saved to:")
print(plot_file)


# =========================================================
# 14. Research interpretation
# =========================================================

print("\n--- Research Interpretation ---")

print(
    "This experiment evaluates whether forecasting "
    "accuracy changes across different housing-market "
    "growth regimes."
)

print(
    "The results can identify periods in which the "
    "machine-learning model is more or less reliable."
)