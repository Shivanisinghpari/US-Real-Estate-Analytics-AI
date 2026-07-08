# US Real Estate Analytics & AI Predictive Pipeline

An automated data engineering pipeline designed to ingest, clean, and process historical United States housing market data for macroeconomic trend analysis and machine learning forecasting models.

## 🚀 Project Overview
This project serves as an end-to-end framework demonstrating data lifecycle management. It dynamically fetches live indicators from the Federal Reserve Economic Data (FRED) system, executes automated data transformations, and exports analytics-ready payloads for Business Intelligence (BI) software and predictive AI modeling.

## ⚙️ Core Technical Features
* **Automated Data Sourcing:** Connects programmatically to live remote CSV endpoints hosted by FRED.
* **Algorithmic Data Transformation:** Normalizes data structures, executes chronological sorting, and resolves missing datasets (`NaN` handling) via Python (Pandas).
* **Feature Engineering:** Calculates Year-over-Year (YoY) real estate growth dynamics and isolates seasonal markers.
* **Statistical Outlier Detection:** Applies a Standard Deviation Z-Score framework ($|Z| > 3$) to flag critical economic market anomalies and structural shifts.

## 📁 Repository Directory Structure
* `data_pipeline.py`: Main execution script containing the data ingestion and transformation logic.
* `data/processed/`: Dedicated workspace containing optimized, cleaned `.csv` datasets optimized for PowerBI integration.

## 🛠️ Tech Stack & Methodology
* **Language:** Python
* **Core Libraries:** Pandas, NumPy, OS
* **Target Applications:** Business Analytics, Urban Informatics, Predictive Data Mining

