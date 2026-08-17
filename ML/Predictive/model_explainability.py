import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error

import matplotlib.pyplot as plt


# =========================================================
# 1. Locate project files
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

input_file = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cleaned_us_housing_market.csv"
)


# =========================================================
# 2. Load historical housing data
# =========================================================

df = pd.read_csv(input_file)

df["Date"] = pd.to_datetime(df["Date"])

df = (
    df.sort_values("Date")
      .reset_index(drop=True)
)


print("\n--- Model Explainability: Data Loaded ---")

print("Dataset shape:", df.shape)

print(
    "Date range:",
    df["Date"].min(),
    "to",
    df["Date"].max()
)


# =========================================================
# 3. Feature Engineering
# =========================================================

df["Time_Index"] = range(len(df))


# Previous quarter price
df["Price_Lag_1"] = (
    df["Median_Price"]
    .shift(1)
)


# Same quarter previous year
df["Price_Lag_4"] = (
    df["Median_Price"]
    .shift(4)
)


# Previous quarter growth
df["Growth_Lag_1"] = (
    df["Median_Price"]
    .pct_change()
    .shift(1)
    * 100
)


# Previous year growth
df["Growth_Lag_4"] = (
    df["Median_Price"]
    .pct_change(4)
    .shift(1)
    * 100
)


# Historical four-quarter average
df["Rolling_Mean_4"] = (
    df["Median_Price"]
    .shift(1)
    .rolling(4)
    .mean()
)


# Target: quarterly growth
df["Target_Growth"] = (
    df["Median_Price"]
    .pct_change()
    * 100
)


# =========================================================
# 4. Define features
# =========================================================

features = [
    "Year",
    "Quarter",
    "Time_Index",
    "Price_Lag_1",
    "Price_Lag_4",
    "Growth_Lag_1",
    "Growth_Lag_4",
    "Rolling_Mean_4"
]


df_model = df[
    ["Date"]
    + features
    + ["Median_Price", "Target_Growth"]
].copy()


df_model = (
    df_model
    .dropna()
    .reset_index(drop=True)
)


print("\n--- Explainability Dataset ---")
print("Shape:", df_model.shape)

print("\nFeatures:")
print(features)


# =========================================================
# 5. Chronological Train/Test Split
# =========================================================

split_index = int(len(df_model) * 0.80)

train = df_model.iloc[:split_index].copy()

test = df_model.iloc[split_index:].copy()


X_train = train[features]

X_test = test[features]

y_train = train["Target_Growth"]

y_test = test["Target_Growth"]


print("\n--- Train/Test Split ---")

print("Training observations:", len(train))

print("Testing observations:", len(test))

print(
    "Training period:",
    train["Date"].min(),
    "to",
    train["Date"].max()
)

print(
    "Testing period:",
    test["Date"].min(),
    "to",
    test["Date"].max()
)


# =========================================================
# 6. Train Random Forest
# =========================================================

model = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    min_samples_leaf=2,
    n_jobs=-1
)


model.fit(
    X_train,
    y_train
)


print("\n--- Random Forest ---")
print("Model training completed.")


# =========================================================
# 7. Test Predictions
# =========================================================

predicted_growth = model.predict(X_test)


# Reconstruct price
previous_prices = (
    test["Price_Lag_1"]
    .values
)


predicted_prices = (
    previous_prices
    * (1 + predicted_growth / 100)
)


actual_prices = (
    test["Median_Price"]
    .values
)


# =========================================================
# 8. Model Performance
# =========================================================

mae = mean_absolute_error(
    actual_prices,
    predicted_prices
)


rmse = np.sqrt(
    mean_squared_error(
        actual_prices,
        predicted_prices
    )
)


mape = np.mean(
    np.abs(
        (actual_prices - predicted_prices)
        / actual_prices
    )
) * 100


print("\n--- Model Performance ---")

print(f"MAE : {mae:.2f}")

print(f"RMSE: {rmse:.2f}")

print(f"MAPE: {mape:.2f}%")


# =========================================================
# 9. Built-in Random Forest Feature Importance
# =========================================================

feature_importance = pd.DataFrame({

    "Feature": features,

    "Importance": model.feature_importances_

})


feature_importance = (
    feature_importance
    .sort_values(
        "Importance",
        ascending=False
    )
    .reset_index(drop=True)
)


print("\n--- Random Forest Feature Importance ---")

print(feature_importance)


# =========================================================
# 10. Save Feature Importance
# =========================================================

output_dir = (
    PROJECT_ROOT
    / "data"
    / "forecasts"
    / "explainability"
)


output_dir.mkdir(
    parents=True,
    exist_ok=True
)


importance_file = (
    output_dir
    / "random_forest_feature_importance.csv"
)


feature_importance.to_csv(
    importance_file,
    index=False
)


print(
    "\nFeature importance saved to:"
)

print(importance_file)


# =========================================================
# 11. Permutation Importance
# =========================================================

print(
    "\n--- Calculating Permutation Importance ---"
)


permutation = permutation_importance(
    model,
    X_test,
    y_test,
    n_repeats=20,
    random_state=42,
    scoring="neg_mean_absolute_error",
    n_jobs=-1
)


permutation_importance_df = pd.DataFrame({

    "Feature": features,

    "Importance_Mean":
        permutation.importances_mean,

    "Importance_Std":
        permutation.importances_std

})


permutation_importance_df = (
    permutation_importance_df
    .sort_values(
        "Importance_Mean",
        ascending=False
    )
    .reset_index(drop=True)
)


print(
    "\n--- Permutation Importance ---"
)

print(permutation_importance_df)


# =========================================================
# 12. Save Permutation Importance
# =========================================================

permutation_file = (
    output_dir
    / "permutation_feature_importance.csv"
)


permutation_importance_df.to_csv(
    permutation_file,
    index=False
)


print(
    "\nPermutation importance saved to:"
)

print(permutation_file)


# =========================================================
# 13. Compare Feature Importance Methods
# =========================================================

importance_comparison = (
    feature_importance
    .merge(
        permutation_importance_df,
        on="Feature",
        how="left"
    )
)


importance_comparison["RF_Rank"] = (
    importance_comparison["Importance"]
    .rank(
        ascending=False,
        method="min"
    )
)


importance_comparison["Permutation_Rank"] = (
    importance_comparison["Importance_Mean"]
    .rank(
        ascending=False,
        method="min"
    )
)


importance_comparison = (
    importance_comparison
    .sort_values("RF_Rank")
    .reset_index(drop=True)
)


print(
    "\n--- Feature Importance Comparison ---"
)

print(importance_comparison)


comparison_file = (
    output_dir
    / "feature_importance_comparison.csv"
)


importance_comparison.to_csv(
    comparison_file,
    index=False
)


print(
    "\nComparison saved to:"
)

print(comparison_file)


# =========================================================
# 14. Feature Importance Visualization
# =========================================================

plt.figure(
    figsize=(10, 6)
)


plt.barh(
    feature_importance["Feature"],
    feature_importance["Importance"]
)


plt.xlabel(
    "Random Forest Importance"
)


plt.ylabel(
    "Feature"
)


plt.title(
    "Random Forest Feature Importance"
)


plt.gca().invert_yaxis()


plt.grid(
    axis="x",
    alpha=0.3
)


plt.tight_layout()


importance_plot = (
    output_dir
    / "random_forest_feature_importance.png"
)


plt.savefig(
    importance_plot,
    dpi=300,
    bbox_inches="tight"
)


plt.show()


print(
    "\nFeature importance visualization saved to:"
)

print(importance_plot)


# =========================================================
# 15. Permutation Importance Visualization
# =========================================================

plt.figure(
    figsize=(10, 6)
)


plt.barh(
    permutation_importance_df["Feature"],
    permutation_importance_df["Importance_Mean"]
)


plt.xlabel(
    "Decrease in Model Performance"
)


plt.ylabel(
    "Feature"
)


plt.title(
    "Permutation Feature Importance"
)


plt.gca().invert_yaxis()


plt.grid(
    axis="x",
    alpha=0.3
)


plt.tight_layout()


permutation_plot = (
    output_dir
    / "permutation_feature_importance.png"
)


plt.savefig(
    permutation_plot,
    dpi=300,
    bbox_inches="tight"
)


plt.show()


print(
    "\nPermutation visualization saved to:"
)

print(permutation_plot)


# =========================================================
# 16. Feature Importance Stability
# =========================================================

importance_comparison["Rank_Difference"] = (
    importance_comparison["RF_Rank"]
    - importance_comparison["Permutation_Rank"]
)


print(
    "\n--- Feature Importance Stability ---"
)

print(
    importance_comparison[
        [
            "Feature",
            "RF_Rank",
            "Permutation_Rank",
            "Rank_Difference"
        ]
    ]
)


# =========================================================
# 17. Save Model Performance
# =========================================================

performance = pd.DataFrame({

    "Model": [
        "Random Forest Growth"
    ],

    "MAE": [
        mae
    ],

    "RMSE": [
        rmse
    ],

    "MAPE": [
        mape
    ]

})


performance_file = (
    output_dir
    / "explainability_model_performance.csv"
)


performance.to_csv(
    performance_file,
    index=False
)


# =========================================================
# 18. Optional SHAP Analysis
# =========================================================

print(
    "\n--- SHAP Explainability ---"
)


try:

    import shap

    print(
        "SHAP detected. Calculating SHAP values..."
    )


    explainer = shap.TreeExplainer(
        model
    )


    shap_values = explainer.shap_values(
        X_test
    )


    # SHAP importance
    shap_importance = pd.DataFrame({

        "Feature": features,

        "Mean_Absolute_SHAP":
            np.abs(shap_values).mean(axis=0)

    })


    shap_importance = (
        shap_importance
        .sort_values(
            "Mean_Absolute_SHAP",
            ascending=False
        )
        .reset_index(drop=True)
    )


    print(
        "\n--- SHAP Feature Importance ---"
    )

    print(shap_importance)


    shap_file = (
        output_dir
        / "shap_feature_importance.csv"
    )


    shap_importance.to_csv(
        shap_file,
        index=False
    )


    # SHAP summary plot
    plt.figure(
        figsize=(10, 7)
    )


    shap.summary_plot(
        shap_values,
        X_test,
        feature_names=features,
        show=False
    )


    plt.tight_layout()


    shap_plot = (
        output_dir
        / "shap_summary_plot.png"
    )


    plt.savefig(
        shap_plot,
        dpi=300,
        bbox_inches="tight"
    )


    plt.show()


    print(
        "\nSHAP analysis saved to:"
    )

    print(shap_file)

    print(shap_plot)


except ImportError:

    print(
        "\nSHAP is not installed."
    )

    print(
        "The Random Forest and permutation "
        "importance analyses were completed."
    )

    print(
        "To enable SHAP later, run:"
    )

    print(
        "pip install shap"
    )


# =========================================================
# 19. Research Summary
# =========================================================

print(
    "\n================================================="
)

print(
    "RESEARCH EXPLAINABILITY SUMMARY"
)

print(
    "================================================="
)

print(
    f"Model MAPE: {mape:.2f}%"
)

print(
    "\nTop Random Forest Features:"
)

print(
    feature_importance.head(5)
)


print(
    "\nTop Permutation Features:"
)

print(
    permutation_importance_df.head(5)
)


print(
    "\nExplainability experiment completed."
)

print(
    "Results directory:"
)

print(output_dir)