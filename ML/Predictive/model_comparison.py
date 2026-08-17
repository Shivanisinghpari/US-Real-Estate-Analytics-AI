import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RESULTS_FOLDER = (
    PROJECT_ROOT
    / "data"
    / "forecasts"
)

REGIME_FOLDER = RESULTS_FOLDER / "regime_analysis"
REGIME_AWARE_FOLDER = RESULTS_FOLDER / "regime_aware"


print("\n" + "=" * 70)
print("FINAL FORECASTING MODEL COMPARISON")
print("=" * 70)

print("\nResults folder:")
print(RESULTS_FOLDER)


# ============================================================
# 2. EXISTING MODEL RESULTS
# ============================================================

result_files = {
    "ARIMA(1,1,1)": "arima_baseline_results.csv",
    "SARIMA(1,1,1)(1,1,1,4)": "sarima_results.csv",
    "Random Forest Baseline": "ml_baseline_results.csv",
    "Random Forest Growth Model": "ml_growth_model_results.csv"
}


all_results = []


# ============================================================
# 3. LOAD EXISTING MODEL RESULTS
# ============================================================

for model_name, filename in result_files.items():

    file_path = RESULTS_FOLDER / filename

    print(f"\nChecking: {filename}")

    if not file_path.exists():
        print("WARNING: File not found.")
        continue

    print("File found.")

    df = pd.read_csv(file_path)

    required_columns = [
        "MAE",
        "RMSE",
        "MAPE"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:

        print(
            f"WARNING: Missing columns: {missing_columns}"
        )

        continue

    row = {
        "Model": model_name,
        "MAE": df["MAE"].iloc[0],
        "RMSE": df["RMSE"].iloc[0],
        "MAPE": df["MAPE"].iloc[0]
    }

    all_results.append(row)


# ============================================================
# 4. FUNCTION TO CALCULATE METRICS FROM PREDICTIONS
# ============================================================

def calculate_prediction_metrics(df):

    actual_column = None
    predicted_column = None

    possible_actual = [
        "Actual_Price",
        "Actual",
        "Actual_Value"
    ]

    possible_predicted = [
        "Predicted_Price",
        "Predicted",
        "Predicted_Value"
    ]

    for column in possible_actual:

        if column in df.columns:
            actual_column = column
            break

    for column in possible_predicted:

        if column in df.columns:
            predicted_column = column
            break

    if actual_column is None or predicted_column is None:

        raise ValueError(
            "Could not identify actual/predicted price columns."
        )

    actual = df[actual_column]
    predicted = df[predicted_column]

    mae = mean_absolute_error(
        actual,
        predicted
    )

    rmse = np.sqrt(
        mean_squared_error(
            actual,
            predicted
        )
    )

    mape = np.mean(
        np.abs(
            (actual - predicted) / actual
        )
    ) * 100

    return mae, rmse, mape


# ============================================================
# 5. ORIGINAL RANDOM FOREST
# ============================================================

original_rf_file = (
    REGIME_FOLDER
    / "regime_predictions.csv"
)

print("\n" + "-" * 70)
print("ORIGINAL RANDOM FOREST")
print("-" * 70)

if original_rf_file.exists():

    original_rf_df = pd.read_csv(
        original_rf_file
    )

    print(
        f"Predictions loaded: "
        f"{len(original_rf_df)} observations"
    )

    mae, rmse, mape = calculate_prediction_metrics(
        original_rf_df
    )

    print(f"MAE  : {mae:.2f}")
    print(f"RMSE : {rmse:.2f}")
    print(f"MAPE : {mape:.2f}%")

    all_results.append({
        "Model": "Random Forest Original",
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape
    })

else:

    print(
        "WARNING: Original Random Forest predictions not found."
    )


# ============================================================
# 6. REGIME-AWARE RANDOM FOREST
# ============================================================

regime_aware_file = (
    REGIME_AWARE_FOLDER
    / "regime_aware_predictions.csv"
)

print("\n" + "-" * 70)
print("REGIME-AWARE RANDOM FOREST")
print("-" * 70)

if regime_aware_file.exists():

    regime_aware_df = pd.read_csv(
        regime_aware_file
    )

    print(
        f"Predictions loaded: "
        f"{len(regime_aware_df)} observations"
    )

    mae, rmse, mape = calculate_prediction_metrics(
        regime_aware_df
    )

    print(f"MAE  : {mae:.2f}")
    print(f"RMSE : {rmse:.2f}")
    print(f"MAPE : {mape:.2f}%")

    all_results.append({
        "Model": "Regime-Aware Random Forest",
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape
    })

else:

    print(
        "WARNING: Regime-aware predictions not found."
    )


# ============================================================
# 7. CHECK RESULTS
# ============================================================

if not all_results:

    raise ValueError(
        "No model evaluation results were found."
    )


# ============================================================
# 8. CREATE FINAL COMPARISON TABLE
# ============================================================

comparison = pd.DataFrame(
    all_results
)


comparison = comparison.sort_values(
    by="MAPE",
    ascending=True
).reset_index(drop=True)


print("\n" + "=" * 70)
print("FINAL MODEL COMPARISON")
print("=" * 70)

print(
    comparison.to_string(
        index=False,
        formatters={
            "MAE": "{:.2f}".format,
            "RMSE": "{:.2f}".format,
            "MAPE": "{:.2f}%".format
        }
    )
)


# ============================================================
# 9. BEST MODEL
# ============================================================

best_model = comparison.iloc[0]

print("\n" + "-" * 70)
print("BEST OVERALL MODEL")
print("-" * 70)

print(
    f"Model: {best_model['Model']}"
)

print(
    f"MAE: {best_model['MAE']:.2f}"
)

print(
    f"RMSE: {best_model['RMSE']:.2f}"
)

print(
    f"MAPE: {best_model['MAPE']:.2f}%"
)


# ============================================================
# 10. SAVE FINAL COMPARISON
# ============================================================

output_file = (
    RESULTS_FOLDER
    / "final_model_comparison.csv"
)

comparison.to_csv(
    output_file,
    index=False
)

print("\nFinal comparison saved to:")
print(output_file)


# ============================================================
# 11. REGIME-LEVEL COMPARISON
# ============================================================

original_regime_file = (
    REGIME_FOLDER
    / "regime_performance.csv"
)

regime_aware_performance_file = (
    REGIME_AWARE_FOLDER
    / "regime_aware_performance.csv"
)


if (
    original_regime_file.exists()
    and regime_aware_performance_file.exists()
):

    print("\n" + "=" * 70)
    print("MARKET REGIME PERFORMANCE COMPARISON")
    print("=" * 70)

    original_regime = pd.read_csv(
        original_regime_file
    )

    regime_aware = pd.read_csv(
        regime_aware_performance_file
    )

    original_regime = original_regime[
        [
            "Market_Regime",
            "Observations",
            "MAE",
            "MAPE"
        ]
    ].copy()

    regime_aware = regime_aware[
        [
            "Market_Regime",
            "Observations",
            "MAE",
            "MAPE"
        ]
    ].copy()

    original_regime = original_regime.rename(
        columns={
            "MAE": "Original_MAE",
            "MAPE": "Original_MAPE"
        }
    )

    regime_aware = regime_aware.rename(
        columns={
            "MAE": "Regime_Aware_MAE",
            "MAPE": "Regime_Aware_MAPE"
        }
    )

    regime_comparison = pd.merge(
        original_regime,
        regime_aware,
        on="Market_Regime",
        how="outer"
    )

    # --------------------------------------------------------
    # Calculate improvement
    # --------------------------------------------------------

    regime_comparison[
        "MAPE_Improvement_Percent"
    ] = (
        (
            regime_comparison["Original_MAPE"]
            - regime_comparison["Regime_Aware_MAPE"]
        )
        /
        regime_comparison["Original_MAPE"]
    ) * 100


    print(
        regime_comparison.to_string(
            index=False,
            formatters={
                "Original_MAE": "{:.2f}".format,
                "Original_MAPE": "{:.2f}%".format,
                "Regime_Aware_MAE": "{:.2f}".format,
                "Regime_Aware_MAPE": "{:.2f}%".format,
                "MAPE_Improvement_Percent": "{:.2f}%".format
            }
        )
    )


    # ========================================================
    # 12. IDENTIFY BEST / WORST REGIME
    # ========================================================

    best_regime = regime_comparison.loc[
        regime_comparison[
            "Regime_Aware_MAPE"
        ].idxmin()
    ]

    largest_improvement = regime_comparison.loc[
        regime_comparison[
            "MAPE_Improvement_Percent"
        ].idxmax()
    ]


    print("\n" + "-" * 70)
    print("REGIME-AWARE MODEL FINDINGS")
    print("-" * 70)

    print(
        "Best regime for regime-aware model:"
    )

    print(
        f"{best_regime['Market_Regime']} "
        f"({best_regime['Regime_Aware_MAPE']:.2f}% MAPE)"
    )

    print(
        "\nLargest MAPE improvement:"
    )

    print(
        f"{largest_improvement['Market_Regime']} "
        f"("
        f"{largest_improvement['MAPE_Improvement_Percent']:.2f}% improvement"
        f")"
    )


    # ========================================================
    # 13. SAVE REGIME COMPARISON
    # ========================================================

    regime_output = (
        RESULTS_FOLDER
        / "final_regime_comparison.csv"
    )

    regime_comparison.to_csv(
        regime_output,
        index=False
    )

    print("\nRegime comparison saved to:")
    print(regime_output)


    # ========================================================
    # 14. REGIME MAPE VISUALIZATION
    # ========================================================

    regimes = regime_comparison[
        "Market_Regime"
    ]

    x = np.arange(
        len(regimes)
    )

    width = 0.35

    plt.figure(
        figsize=(11, 6)
    )

    plt.bar(
        x - width / 2,
        regime_comparison["Original_MAPE"],
        width,
        label="Original Random Forest"
    )

    plt.bar(
        x + width / 2,
        regime_comparison["Regime_Aware_MAPE"],
        width,
        label="Regime-Aware Random Forest"
    )

    plt.xticks(
        x,
        regimes,
        rotation=20
    )

    plt.ylabel(
        "MAPE (%)"
    )

    plt.xlabel(
        "Market Regime"
    )

    plt.title(
        "Original vs Regime-Aware Random Forest by Market Regime"
    )

    plt.legend()

    plt.grid(
        axis="y",
        alpha=0.3
    )

    plt.tight_layout()


    regime_plot = (
        REGIME_AWARE_FOLDER
        / "original_vs_regime_aware_mape.png"
    )

    plt.savefig(
        regime_plot,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    print("\nRegime comparison visualization saved to:")
    print(regime_plot)


else:

    print(
        "\nWARNING: Regime performance files not found."
    )


# ============================================================
# 15. RESEARCH INTERPRETATION
# ============================================================

print("\n" + "=" * 70)
print("RESEARCH INTERPRETATION")
print("=" * 70)

print(
    """
The final comparison evaluates whether incorporating
historical market-regime information improves housing-price
forecasting performance.

The analysis considers both overall forecasting accuracy
and performance under individual market regimes.

A regime-aware model should not be considered superior
solely on the basis of overall MAPE. Particular attention
should be given to whether regime-aware modeling improves
forecast reliability during rapidly changing market
conditions.

The results can therefore distinguish between:

1. Overall predictive accuracy
2. Regime-specific predictive accuracy
3. Model robustness under changing market conditions
"""
)