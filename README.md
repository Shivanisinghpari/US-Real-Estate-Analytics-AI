# US Real Estate Analytics & AI Predictive Pipeline

An automated data engineering pipeline designed to ingest, clean, and process historical United States housing market data for macroeconomic trend analysis and machine learning forecasting models.

## 🚀 Project Overview
This project serves as an end-to-end framework demonstrating data lifecycle management. It dynamically fetches live indicators from the Federal Reserve Economic Data (FRED) system, executes automated data transformations, and exports analytics-ready payloads for Business Intelligence (BI) software and predictive AI modeling.

## ⚙️ Core Technical Features
* **Automated Data Sourcing:** Connects programmatically to live remote CSV endpoints hosted by FRED.
- **Algorithmic Data Transformation:** Normalizes data structures, converts date and numeric fields, and handles missing values (`NaN`) using Python (Pandas).
* **Feature Engineering:** Calculates Year-over-Year (YoY) real estate growth dynamics and isolates seasonal markers.
* **Statistical Outlier Detection:** Applies a Standard Deviation Z-Score framework ($|Z| > 3$) to flag critical economic market anomalies and structural shifts.

## 📁 Repository Directory Structure
* `data_pipeline.py`: Main execution script containing the data ingestion and transformation logic.
- `data/processed/`: Contains cleaned CSV data and the SQLite database used for analytics and Power BI integration.

## 🛠️ Tech Stack & Methodology
* **Language:** Python
* **Core Libraries:** Pandas, NumPy, OS
* **Target Applications:** Business Analytics, Urban Informatics, Predictive Data Mining

## 📊 Power BI Dashboard

The project includes an interactive Power BI dashboard for exploring long-term U.S. housing market trends from 1963–2026.

### Dashboard Highlights

* **Highest Median House Price:** 443K
* **Average YoY Price Growth:** 5.29%
* **Total Quarterly Records:** 254
* Interactive **Year slicer** for filtering the analysis
* Long-term median house price trend analysis
* Year-over-Year housing price growth analysis
* Annual average median house price comparison

### Dashboard Preview

![US Real Estate Analytics Dashboard](powerbi/US_Real_Estate_Analytics_Dashboard.png)

The Power BI source file is available here:

[`US_Real_Estate_Analytics_Dashboard.pbix`](powerbi/US_Real_Estate_Analytics_Dashboard.pbix)

## 🔄 Current Development Roadmap

### Completed

* Automated data ingestion and cleaning
* Feature engineering and YoY growth calculation
* SQLite database pipeline
* SQL validation and analytical queries
* Interactive Power BI dashboard
* Dashboard screenshot and Power BI source file

### Upcoming

* **Machine Learning Forecasting:** Build a model for future housing-market trend prediction.
* **AI Prediction Automation:** Develop an automated process for generating and delivering prediction results.
* **Final Integration & Documentation:** Polish the complete pipeline, improve the project homepage, and document the end-to-end workflow.

* ## Research Findings

The forecasting experiments evaluate traditional statistical models,
machine-learning models, growth-aware features, and market-regime-aware
modeling for US housing price forecasting.

### Overall Model Performance

The Random Forest Growth Model achieved the lowest overall MAPE of
2.42%, outperforming the original Random Forest model (2.71%) and
substantially outperforming the statistical ARIMA and SARIMA benchmarks.

| Model | MAE | RMSE | MAPE |
|---|---:|---:|---:|
| Random Forest Growth Model | 8,676.12 | 10,729.52 | 2.42% |
| Random Forest Original | 7,674.84 | 9,911.15 | 2.71% |
| Regime-Aware Random Forest | 7,690.20 | 9,936.54 | 2.72% |
| SARIMA(1,1,1)(1,1,1,4) | 36,308.37 | 47,377.76 | 9.26% |
| ARIMA(1,1,1) | 59,836.50 | 71,855.06 | 15.55% |
| Random Forest Baseline | 92,126.11 | 107,026.55 | 24.08% |

MAPE was used as the primary model-selection metric because it provides
an interpretable percentage-based measure of forecasting error.

### Regime-Aware Forecasting

The regime-aware Random Forest was evaluated under four historical
market conditions:

- Declining
- Moderate Growth
- Rapid Growth
- Stable

The regime-aware model produced its strongest performance during Rapid
Growth periods, achieving a MAPE of 1.71%.

Compared with the original Random Forest, this represents a 65.30%
reduction in MAPE during Rapid Growth conditions.

This is particularly important because rapidly changing markets are
more difficult to forecast using a single global model.

### Research Interpretation

The results indicate that market-regime information can improve
forecasting reliability under specific market conditions.

However, the regime-aware model does not outperform the original model
across every regime. Its performance improves substantially during
Rapid Growth and Moderate Growth conditions, but deteriorates during
Declining and Stable conditions.

Therefore, regime-aware modeling should not be interpreted as a
universally superior forecasting approach. Instead, the experiment
provides evidence that explicitly modeling market conditions can improve
forecasting under rapidly changing market environments.

### Explainability

Model explainability analysis was performed using:

- Random Forest feature importance
- Permutation feature importance
- SHAP analysis

Growth_Lag_1 was consistently identified as the most important
predictive feature.

The SHAP analysis showed the following ranking of feature influence:

1. Growth_Lag_1
2. Price_Lag_1
3. Price_Lag_4
4. Growth_Lag_4
5. Time_Index
6. Quarter
7. Rolling_Mean_4
8. Year

The agreement between multiple explainability methods provides evidence
that recent housing-market growth and lagged price information play an
important role in the model's predictions.

### Key Research Contribution

The project demonstrates a complete forecasting pipeline combining:

1. Automated data acquisition
2. Data cleaning and preprocessing
3. Feature engineering
4. Statistical forecasting
5. Machine-learning forecasting
6. Walk-forward validation
7. Market-regime detection
8. Regime-aware forecasting
9. Model explainability
10. Comparative model evaluation

The main research finding is that incorporating market-regime information
can substantially improve forecasting performance during rapidly
changing growth periods, although the benefit is regime-dependent.
