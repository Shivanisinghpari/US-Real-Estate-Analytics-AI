import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.metrics import mean_absolute_error

# ---------------------------------------------------------
# 1. Locate project
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

results_file = (
    PROJECT_ROOT
    / "data"
    / "forecasts"
    / "benchmark_fold_results.csv"
)

output_file = (
    PROJECT_ROOT
    / "data"
    / "forecasts"
    / "model_significance_comparison.csv"
)

# ---------------------------------------------------------
# 2. Load fold-level benchmark results
# ---------------------------------------------------------

df = pd.read_csv(results_file)

print("\n--- Model Significance Analysis ---")
print("Results loaded from:")
print(results_file)

print("\nModels found:")
print(df["Model"].unique())

# ---------------------------------------------------------
# 3. Select Random Forest and Historical Mean
# ---------------------------------------------------------

rf = df[
    df["Model"] == "Random Forest Growth"
].copy()

historical_mean = df[
    df["Model"] == "Historical Mean Growth"
].copy()

rf = rf.sort_values("Fold")
historical_mean = historical_mean.sort_values("Fold")

# ---------------------------------------------------------
# 4. Compare fold-level errors
# ---------------------------------------------------------

comparison = pd.DataFrame({
    "Fold": rf["Fold"].values,

    "RF_MAE": rf["MAE"].values,

    "Historical_Mean_MAE":
        historical_mean["MAE"].values,

    "RF_MAPE": rf["MAPE"].values,

    "Historical_Mean_MAPE":
        historical_mean["MAPE"].values
})

comparison["MAE_Difference"] = (
    comparison["RF_MAE"]
    - comparison["Historical_Mean_MAE"]
)

comparison["MAPE_Difference"] = (
    comparison["RF_MAPE"]
    - comparison["Historical_Mean_MAPE"]
)

print("\n--- Fold-Level Comparison ---")
print(comparison.to_string(index=False))

# ---------------------------------------------------------
# 5. Calculate average differences
# ---------------------------------------------------------

mean_mae_difference = (
    comparison["MAE_Difference"].mean()
)

mean_mape_difference = (
    comparison["MAPE_Difference"].mean()
)

print("\n--- Average Error Difference ---")

print(
    f"Mean MAE difference "
    f"(RF - Historical Mean): "
    f"{mean_mae_difference:.2f}"
)

print(
    f"Mean MAPE difference "
    f"(RF - Historical Mean): "
    f"{mean_mape_difference:.4f} percentage points"
)

# ---------------------------------------------------------
# 6. Count which model wins each fold
# ---------------------------------------------------------

rf_mae_wins = (
    comparison["RF_MAE"]
    < comparison["Historical_Mean_MAE"]
).sum()

historical_mae_wins = (
    comparison["Historical_Mean_MAE"]
    < comparison["RF_MAE"]
).sum()

rf_mape_wins = (
    comparison["RF_MAPE"]
    < comparison["Historical_Mean_MAPE"]
).sum()

historical_mape_wins = (
    comparison["Historical_Mean_MAPE"]
    < comparison["RF_MAPE"]
).sum()

print("\n--- Fold-Level Wins ---")

print(
    "Random Forest MAE wins:",
    rf_mae_wins
)

print(
    "Historical Mean MAE wins:",
    historical_mae_wins
)

print(
    "Random Forest MAPE wins:",
    rf_mape_wins
)

print(
    "Historical Mean MAPE wins:",
    historical_mape_wins
)

# ---------------------------------------------------------
# 7. Paired error comparison
# ---------------------------------------------------------

mae_difference = comparison[
    "MAE_Difference"
].values

mape_difference = comparison[
    "MAPE_Difference"
].values

# Mean and standard deviation of differences

mae_diff_mean = np.mean(mae_difference)
mae_diff_std = np.std(
    mae_difference,
    ddof=1
)

mape_diff_mean = np.mean(mape_difference)
mape_diff_std = np.std(
    mape_difference,
    ddof=1
)

print("\n--- Paired Error Statistics ---")

print(
    f"MAE difference mean: "
    f"{mae_diff_mean:.2f}"
)

print(
    f"MAE difference std: "
    f"{mae_diff_std:.2f}"
)

print(
    f"MAPE difference mean: "
    f"{mape_diff_mean:.4f}"
)

print(
    f"MAPE difference std: "
    f"{mape_diff_std:.4f}"
)

# ---------------------------------------------------------
# 8. Effect size
# ---------------------------------------------------------

if mae_diff_std != 0:

    mae_effect_size = (
        mae_diff_mean
        / mae_diff_std
    )

else:

    mae_effect_size = 0


if mape_diff_std != 0:

    mape_effect_size = (
        mape_diff_mean
        / mape_diff_std
    )

else:

    mape_effect_size = 0


print("\n--- Effect Size ---")

print(
    f"MAE standardized difference: "
    f"{mae_effect_size:.3f}"
)

print(
    f"MAPE standardized difference: "
    f"{mape_effect_size:.3f}"
)

# ---------------------------------------------------------
# 9. Research interpretation
# ---------------------------------------------------------

if mean_mape_difference < 0:

    conclusion = (
        "Random Forest has lower average MAPE "
        "than the Historical Mean benchmark."
    )

elif mean_mape_difference > 0:

    conclusion = (
        "Historical Mean has lower average MAPE "
        "than the Random Forest model."
    )

else:

    conclusion = (
        "Both models have identical average MAPE."
    )

print("\n--- Research Interpretation ---")
print(conclusion)

print(
    "\nThe fold-level comparison measures whether "
    "the Random Forest provides consistent predictive "
    "advantages over a simple historical baseline."
)

# ---------------------------------------------------------
# 10. Save comparison
# ---------------------------------------------------------

output_file.parent.mkdir(
    parents=True,
    exist_ok=True
)

comparison.to_csv(
    output_file,
    index=False
)

print("\n--- Significance Analysis Saved ---")
print("Results saved to:")
print(output_file)