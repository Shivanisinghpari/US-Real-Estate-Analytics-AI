"""
Final Research Results Visualization
=====================================

Purpose:
Create publication-style visualizations summarizing the final
U.S. housing-price forecasting research results.

Primary evaluation:
Walk-forward validation is treated as the main measure of
model generalization.

Additional analyses:
1. Overall model comparison
2. Market-regime performance
3. Feature importance
4. Future forecast with prediction interval
"""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FORECAST_DIR = PROJECT_ROOT / "data" / "forecasts"
EXPLAINABILITY_DIR = FORECAST_DIR / "explainability"
OUTPUT_DIR = FORECAST_DIR / "research_results"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# GLOBAL PLOT SETTINGS
# ============================================================

plt.rcParams.update({
    "figure.figsize": (11, 7),
    "font.size": 11,
    "axes.titlesize": 16,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})


# ============================================================
# 1. WALK-FORWARD MODEL COMPARISON
# ============================================================

def create_model_comparison():

    file_path = FORECAST_DIR / "walk_forward_model_comparison.csv"

    df = pd.read_csv(file_path)

    df = df.sort_values("MAPE")

    plt.figure(figsize=(11, 7))

    bars = plt.bar(
        df["Model"],
        df["MAPE"]
    )

    plt.ylabel("Mean MAPE (%)")
    plt.xlabel("Forecasting Model")
    plt.title(
        "Walk-Forward Forecasting Performance",
        fontweight="bold"
    )

    plt.xticks(rotation=20, ha="right")

    for bar, value in zip(bars, df["MAPE"]):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.2f}%",
            ha="center",
            va="bottom"
        )

    plt.grid(axis="y", alpha=0.25)

    plt.tight_layout()

    output = OUTPUT_DIR / "01_walk_forward_model_comparison.png"

    plt.savefig(output, dpi=300, bbox_inches="tight")

    plt.close()

    print(f"Saved: {output}")


# ============================================================
# 2. MARKET REGIME PERFORMANCE
# ============================================================

def create_regime_comparison():

    file_path = FORECAST_DIR / "final_regime_comparison.csv"

    df = pd.read_csv(file_path)

    x = range(len(df))
    width = 0.35

    plt.figure(figsize=(11, 7))

    plt.bar(
        [i - width / 2 for i in x],
        df["Original_MAPE"],
        width=width,
        label="Original Random Forest"
    )

    plt.bar(
        [i + width / 2 for i in x],
        df["Regime_Aware_MAPE"],
        width=width,
        label="Regime-Aware Random Forest"
    )

    plt.xticks(
        list(x),
        df["Market_Regime"]
    )

    plt.ylabel("MAPE (%)")
    plt.xlabel("Market Regime")

    plt.title(
        "Forecasting Performance by Market Regime",
        fontweight="bold"
    )

    plt.legend()

    plt.grid(axis="y", alpha=0.25)

    plt.tight_layout()

    output = OUTPUT_DIR / "02_market_regime_comparison.png"

    plt.savefig(output, dpi=300, bbox_inches="tight")

    plt.close()

    print(f"Saved: {output}")


# ============================================================
# 3. FEATURE IMPORTANCE
# ============================================================

def create_feature_importance():

    file_path = (
        EXPLAINABILITY_DIR /
        "random_forest_feature_importance.csv"
    )

    df = pd.read_csv(file_path)

    df = df.sort_values(
        "Importance",
        ascending=True
    )

    plt.figure(figsize=(11, 7))

    plt.barh(
        df["Feature"],
        df["Importance"]
    )

    plt.xlabel("Random Forest Importance")
    plt.ylabel("Feature")

    plt.title(
        "Random Forest Feature Importance",
        fontweight="bold"
    )

    plt.grid(axis="x", alpha=0.25)

    plt.tight_layout()

    output = OUTPUT_DIR / "03_feature_importance.png"

    plt.savefig(output, dpi=300, bbox_inches="tight")

    plt.close()

    print(f"Saved: {output}")


# ============================================================
# 4. FUTURE FORECAST + 95% INTERVAL
# ============================================================

def create_future_forecast():

    forecast_file = (
        FORECAST_DIR /
        "future_forecast_with_uncertainty.csv"
    )

    historical_file = (
        PROJECT_ROOT /
        "data" /
        "processed" /
        "cleaned_us_housing_market.csv"
    )

    forecast = pd.read_csv(forecast_file)

    historical = pd.read_csv(historical_file)

    historical["Date"] = pd.to_datetime(historical["Date"])
    forecast["Date"] = pd.to_datetime(forecast["Date"])

    plt.figure(figsize=(12, 7))

    plt.plot(
        historical["Date"],
        historical["Median_Price"],
        label="Historical Median Price",
        linewidth=2
    )

    plt.plot(
        forecast["Date"],
        forecast["Predicted_Price"],
        linestyle="--",
        linewidth=2,
        label="ML Forecast"
    )

    plt.fill_between(
        forecast["Date"],
        forecast["Predicted_Price_Lower_95"],
        forecast["Predicted_Price_Upper_95"],
        alpha=0.20,
        label="95% Prediction Interval"
    )

    plt.axvline(
        historical["Date"].max(),
        linestyle=":",
        linewidth=2
    )

    plt.xlabel("Date")
    plt.ylabel("Median House Price ($)")

    plt.title(
        "U.S. Median House Price: Historical Data and "
        "8-Quarter ML Forecast",
        fontweight="bold"
    )

    plt.legend()

    plt.grid(alpha=0.25)

    plt.tight_layout()

    output = OUTPUT_DIR / "04_future_forecast_uncertainty.png"

    plt.savefig(output, dpi=300, bbox_inches="tight")

    plt.close()

    print(f"Saved: {output}")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("FINAL RESEARCH RESULTS VISUALIZATION")
    print("=" * 70)

    create_model_comparison()

    create_regime_comparison()

    create_feature_importance()

    create_future_forecast()

    print("\nAll research visualizations generated successfully.")

    print(f"\nOutput directory:")
    print(OUTPUT_DIR)