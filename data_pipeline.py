import os
import pandas as pd
import numpy as np

def fetch_real_estate_data():
    print("LOG: Fetching source data from FRED gateway...")
    url = "https://stlouisfed.org"
    try:
        df = pd.read_csv(url)
        print(f"SUCCESS: Retrieved {len(df)} source records.")
        return df
    except Exception as e:
        print(f"ERROR: Data download failed. Details: {e}")
        return None

def clean_and_transform(df):
    print("LOG: Starting data cleaning and feature engineering...")
    df.columns = ['Date', 'Median_Price']
    df['Date'] = pd.to_datetime(df['Date'])
    df['Median_Price'] = pd.to_numeric(df['Median_Price'], errors='coerce')
    
    # Drop null rows
    df = df.dropna(subset=['Median_Price'])
    
    # Generate time components for dashboard indexing
    df['Year'] = df['Date'].dt.year
    df['Quarter'] = df['Date'].dt.quarter
    
    # Calculate YoY variance
    df['Price_YoY_Growth_Pct'] = df['Median_Price'].pct_change(periods=4) * 100
    
    # Outlier detection using standard deviation threshold
    price_mean = df['Median_Price'].mean()
    price_std = df['Median_Price'].std()
    df['Is_Outlier'] = np.where(np.abs((df['Median_Price'] - price_mean) / price_std) > 3, 1, 0)
    
    print("LOG: Transformation complete.")
    return df

if __name__ == "__main__":
    raw_data = fetch_real_estate_data()
    if raw_data is not None:
        processed_data = clean_and_transform(raw_data)
        os.makedirs("data/processed", exist_ok=True)
        output_path = "data/processed/cleaned_us_housing_market.csv"
        processed_data.to_csv(output_path, index=False)
        print(f"LOG: Output saved to {output_path}")
