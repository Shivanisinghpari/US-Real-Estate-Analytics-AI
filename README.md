# US Real Estate Analytics & AI Predictive Pipeline

An end-to-end data engineering, forecasting, and explainable AI pipeline for analyzing and predicting long-term United States housing market trends.

The project integrates automated data acquisition, data preprocessing, feature engineering, statistical forecasting, machine-learning models, market-regime analysis, walk-forward validation, and model explainability.

The primary objective is to investigate whether historical housing-market dynamics, growth patterns, and market-regime information can improve forecasting accuracy and reliability under changing market conditions.

---

## 🚀 Project Overview

This project provides an end-to-end framework for the analysis and forecasting of U.S. housing-market data.

Historical quarterly housing-market data is automatically processed through a data engineering pipeline and transformed into analytics-ready datasets for SQL, Power BI, statistical forecasting, and machine-learning experiments.

The forecasting framework evaluates multiple approaches, including:

* ARIMA
* SARIMA
* Random Forest
* Growth-aware Random Forest
* Regime-aware Random Forest

The project also incorporates explainability techniques to investigate which historical market features contribute most strongly to model predictions.

---

## 🎯 Research Objective

The central research question is:

> **Can incorporating historical growth patterns and market-regime information improve U.S. housing-price forecasting accuracy, particularly during rapidly changing market conditions?**

The project investigates this question through:

1. Feature engineering based on historical housing-price behavior
2. Statistical forecasting benchmarks
3. Machine-learning forecasting
4. Growth-aware modeling
5. Market-regime detection
6. Regime-aware forecasting
7. Walk-forward validation
8. Model comparison
9. Feature-importance analysis
10. SHAP-based explainability

---

## ⚙️ Core Technical Features

### Automated Data Sourcing

The pipeline programmatically retrieves historical U.S. housing-market indicators from the Federal Reserve Economic Data (FRED) system.

### Data Cleaning & Transformation

Raw data is transformed into an analytics-ready structure using Python and Pandas.

Processing includes:

* Date conversion
* Numeric conversion
* Missing-value handling
* Data validation
* Dataset restructuring

### Feature Engineering

The forecasting pipeline creates temporal and historical-market features including:

* Year
* Quarter
* Time Index
* Price Lag 1
* Price Lag 4
* Growth Lag 1
* Growth Lag 4
* Rolling Mean 4
* Market Regime

These features allow the machine-learning models to capture both short-term and seasonal housing-market dynamics.

### Statistical Outlier Detection

A Standard Deviation Z-Score framework is used to identify potentially anomalous observations and structural changes in historical market behavior.

---

# 🤖 Forecasting & Machine Learning

The project evaluates multiple forecasting approaches to establish meaningful performance benchmarks.

## Statistical Models

### ARIMA

An ARIMA(1,1,1) model is used as a traditional time-series forecasting benchmark.

### SARIMA

A seasonal SARIMA model,

`SARIMA(1,1,1)(1,1,1,4)`

is evaluated to account for quarterly seasonal behavior.

---

## Machine-Learning Models

### Random Forest Baseline

A basic Random Forest model provides an initial machine-learning benchmark.

### Original Random Forest

A more developed Random Forest model incorporates historical price and growth features.

### Growth-Aware Random Forest

The growth-aware model explicitly incorporates historical housing-price growth dynamics into the feature set.

### Regime-Aware Random Forest

The regime-aware model incorporates historical market conditions into the forecasting process.

Four market regimes are analyzed:

* Declining
* Moderate Growth
* Rapid Growth
* Stable

---

# 🔄 Walk-Forward Validation

Because housing-price data is time-dependent, the project uses walk-forward validation rather than randomly shuffling observations.

Each validation fold trains the model on historical observations and evaluates it on a subsequent unseen time period.

This approach helps reduce temporal leakage and provides a more realistic evaluation of forecasting performance.

The regime-aware experiment used five sequential validation folds covering historical periods through 2026.

---

# 📊 Final Model Comparison

The following results were obtained from the completed forecasting experiments.

| Model                      |          MAE |         RMSE |      MAPE |
| -------------------------- | -----------: | -----------: | --------: |
| Random Forest Growth Model |     8,676.12 |    10,729.52 | **2.42%** |
| Random Forest Original     | **7,674.84** | **9,911.15** |     2.71% |
| Regime-Aware Random Forest |     7,690.20 |     9,936.54 |     2.72% |
| SARIMA(1,1,1)(1,1,1,4)     |    36,308.37 |    47,377.76 |     9.26% |
| ARIMA(1,1,1)               |    59,836.50 |    71,855.06 |    15.55% |
| Random Forest Baseline     |    92,126.11 |   107,026.55 |    24.08% |

MAPE was used as the primary model-selection metric because it provides an interpretable percentage-based measure of forecasting error.

### Overall Finding

The **Random Forest Growth Model achieved the lowest overall MAPE of 2.42%**.

The original Random Forest achieved the lowest MAE and RMSE among the Random Forest variants, while the growth-aware model achieved the lowest percentage forecasting error.

Therefore, the growth-aware model is considered the best overall model when MAPE is used as the primary selection criterion.

---

# 📈 Regime-Aware Forecasting

The regime-aware Random Forest was evaluated separately across four historical market conditions.

| Market Regime   | Original RF MAPE | Regime-Aware RF MAPE | MAPE Improvement |
| --------------- | ---------------: | -------------------: | ---------------: |
| Declining       |            3.25% |                3.55% |           -9.35% |
| Moderate Growth |            2.88% |                2.42% |          +15.91% |
| Rapid Growth    |            4.94% |            **1.71%** |      **+65.30%** |
| Stable          |            0.82% |                2.15% |         -162.76% |

## Key Finding

The most significant result occurs during **Rapid Growth** periods.

The original Random Forest achieved a MAPE of **4.94%**, while the regime-aware model reduced this to **1.71%**.

This represents a **65.30% reduction in MAPE**.

The result suggests that explicitly incorporating market-regime information can substantially improve forecasting during rapidly changing housing-market conditions.

However, the regime-aware model does not outperform the original model in every regime.

Its performance:

* Improves during Moderate Growth
* Improves substantially during Rapid Growth
* Deteriorates during Declining periods
* Deteriorates during Stable periods

Therefore, the results do **not** support the conclusion that regime-aware modeling is universally superior.

Instead, they provide evidence that its usefulness is **regime-dependent**, with the strongest benefit occurring during rapidly changing growth periods.

---

# 🔍 Model Explainability

Model explainability was performed to investigate which features contribute most strongly to the Random Forest predictions.

Three complementary approaches were evaluated:

1. Random Forest feature importance
2. Permutation feature importance
3. SHAP analysis

## Random Forest Feature Importance

The strongest features identified by the Random Forest were:

| Rank | Feature        | Importance |
| ---: | -------------- | ---------: |
|    1 | Growth_Lag_1   |     0.2698 |
|    2 | Growth_Lag_4   |     0.2357 |
|    3 | Quarter        |     0.1066 |
|    4 | Price_Lag_1    |     0.0946 |
|    5 | Price_Lag_4    |     0.0852 |
|    6 | Time_Index     |     0.0781 |
|    7 | Rolling_Mean_4 |     0.0741 |
|    8 | Year           |     0.0558 |

The results indicate that recent housing-market growth dynamics are particularly important to the Random Forest model.

---

## Permutation Feature Importance

Permutation analysis provided additional evidence that `Growth_Lag_1` is the most influential feature.

The measured importance of the remaining features was substantially smaller, while some features produced slightly negative permutation scores.

Negative permutation importance does not necessarily mean that the feature is inherently harmful. It can occur when the feature provides little independent predictive information or when correlated features allow the model to recover similar information through other variables.

---

## SHAP Explainability

SHAP analysis was subsequently enabled to provide a more detailed explanation of feature contributions.

The resulting mean absolute SHAP importance was:

| Rank | Feature        | Mean Absolute SHAP |
| ---: | -------------- | -----------------: |
|    1 | Growth_Lag_1   |             0.6372 |
|    2 | Price_Lag_1    |             0.5900 |
|    3 | Price_Lag_4    |             0.4797 |
|    4 | Growth_Lag_4   |             0.3798 |
|    5 | Time_Index     |             0.3340 |
|    6 | Quarter        |             0.3238 |
|    7 | Rolling_Mean_4 |             0.2448 |
|    8 | Year           |             0.2184 |

The SHAP results further emphasize the importance of recent growth and lagged housing-price information.

The combination of tree-based feature importance, permutation analysis, and SHAP provides multiple perspectives on model behavior rather than relying on a single explainability technique.

---

# 📊 Power BI Dashboard

The project includes an interactive Power BI dashboard for exploring long-term U.S. housing-market trends from **1963–2026**.

## Dashboard Highlights

- Highest Median House Price: **$443K**
- Average YoY Price Growth: **5.29%**
- Total Quarterly Records: **254**
- Interactive **Year slicer**
- Long-term median house-price trend
- Year-over-Year housing-price growth analysis
- Annual average median house-price comparison

### Dashboard Preview

![US Real Estate Analytics Dashboard](./powerbi/US_Real_Estate_Analytics_Dashboard.png)

### Power BI Source File

The interactive Power BI source file is available here:

[**US_Real_Estate_Analytics_Dashboard.pbix**](./powerbi/US_Real_Estate_Analytics_Dashboard.pbix)

---

# 📁 Repository Structure

```text
US-Real-Estate-Analytics-AI/
│
├── data/
│   ├── processed/
│   └── forecasts/
│       ├── explainability/
│       ├── regime_aware/
│       └── final_model_comparison.csv
│
├── ML/
│   ├── Forecasting/
│   │   ├── arima_baseline.py
│   │   ├── sarima_model.py
│   │   ├── model_evaluation.py
│   │   ├── forecast_visualization.py
│   │   └── validate_forecast.py
│   │
│   └── Predictive/
│       ├── ablation_study.py
│       ├── benchmark_growth_models.py
│       ├── feature_engineering.py
│       ├── future_forecast.py
│       ├── ml_baseline.py
│       ├── ml_growth_model.py
│       ├── model_comparison.py
│       ├── model_significance_test.py
│       ├── regime_analysis.py
│       ├── regime_aware_model.py
│       ├── residual_analysis.py
│       └── walk_forward_validation_v1.py
│
├── data_pipeline.py
├── powerbi/
│   └── US_Real_Estate_Analytics_Dashboard.pbix
│
└── README.md
```

---

# 🛠️ Tech Stack

### Programming

* Python

### Data Processing

* Pandas
* NumPy

### Machine Learning

* Scikit-learn
* Random Forest
* Feature Engineering
* Walk-Forward Validation

### Statistical Forecasting

* ARIMA
* SARIMA

### Explainable AI

* SHAP
* Permutation Feature Importance
* Random Forest Feature Importance

### Database & Analytics

* SQLite
* SQL

### Business Intelligence

* Microsoft Power BI

### Data Source

* Federal Reserve Economic Data (FRED)

---

# 🔬 Research Contribution

The project demonstrates an integrated forecasting framework combining:

1. Automated data acquisition
2. Data cleaning and preprocessing
3. Feature engineering
4. Statistical forecasting
5. Machine-learning forecasting
6. Growth-aware modeling
7. Market-regime detection
8. Regime-aware forecasting
9. Walk-forward validation
10. Model comparison
11. Feature importance analysis
12. Permutation importance
13. SHAP explainability
14. Business intelligence visualization

The primary research finding is that **market-regime information can substantially improve forecasting performance during rapidly changing growth periods**, although the benefit varies across market conditions.

The **65.30% reduction in MAPE during Rapid Growth periods** provides the strongest evidence supporting the regime-aware modeling hypothesis.

At the same time, the deterioration in Stable and Declining regimes demonstrates that regime-aware modeling introduces a trade-off and should be evaluated according to the specific forecasting conditions rather than being treated as universally superior.

---

# 📌 Research Interpretation

The experiments suggest three important conclusions:

### 1. Growth information matters

Recent housing-market growth is consistently among the strongest predictors of future housing prices.

### 2. Regime information matters under changing conditions

Market-regime information is particularly valuable during periods of rapid housing-price growth, where the original Random Forest experienced substantially higher forecasting error.

### 3. Model performance is regime-dependent

No single modeling strategy performs best under every historical market condition.

This suggests that future research could investigate **adaptive or dynamic model selection**, where the forecasting strategy changes according to the detected market regime.

---

# 🚀 Future Research Directions

The current results provide several directions for further investigation:

* Dynamic model selection based on detected market regime
* Ensemble forecasting across statistical and machine-learning models
* Hyperparameter optimization
* Additional macroeconomic variables from FRED
* Interest-rate and inflation features
* Housing-market supply and demand indicators
* Advanced time-series models
* Gradient boosting models such as XGBoost
* Temporal cross-validation strategies
* Confidence and prediction intervals
* More robust regime-detection techniques
* Automated future forecasting and prediction delivery
* Comparative evaluation across different U.S. housing indicators

---

# 📈 Project Status

## Completed

* Automated data ingestion
* Data cleaning and preprocessing
* Feature engineering
* YoY growth calculation
* SQLite database pipeline
* SQL validation and analytical queries
* Power BI dashboard
* Dashboard screenshot and Power BI source file
* ARIMA forecasting
* SARIMA forecasting
* Random Forest baseline
* Original Random Forest forecasting
* Growth-aware Random Forest
* Walk-forward validation
* Market-regime analysis
* Regime-aware Random Forest
* Model comparison
* Residual analysis
* Random Forest feature importance
* Permutation feature importance
* SHAP explainability
* Research interpretation

## Current Focus

* Final forecasting pipeline integration
* Future prediction automation
* Final visualization and documentation
* Research-oriented presentation of results

---

# 🧠 Overall Project Outcome

This project evolved from a data-engineering and business-intelligence pipeline into an experimental machine-learning forecasting framework.

The final system combines **data engineering, predictive analytics, time-series forecasting, machine learning, market-regime analysis, and explainable AI** to investigate U.S. housing-market behavior.

The results demonstrate that incorporating historical growth dynamics and market conditions can provide meaningful improvements in forecasting accuracy under specific market regimes, particularly during rapid growth periods.
