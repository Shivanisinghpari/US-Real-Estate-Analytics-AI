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
