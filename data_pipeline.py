import os
import pandas as pd
import numpy as np

def fetch_real_estate_data():
    """Downloads live US Real Estate Market Data from the Federal Reserve website."""
    print("📡 Connecting to data source...")
    url = "https://stlouisfed.org"
    try:
        df = pd.read_csv(url)
        print(f"✅ Successfully downloaded {len(df)} historical records.")
        return df
    except Exception as e:
        print(f"❌ Error downloading data: {e}")
        return None

def clean_and_transform(df):
    """Cleans the data, fixes dates, and checks for market anomalies."""
    print("🧹 Starting data cleaning pipeline...")
    df.columns = ['Date', 'Median_Price']
    df['Date'] = pd.to_datetime(df['Date'])
    df['Median_Price'] = pd.to_numeric(df['Median_Price'], errors='coerce')
    
    # Remove missing data
    df = df.dropna(subset=['Median_Price'])
    
    # Create time filters for PowerBI sorting
    df['Year'] = df['Date'].dt.year
    df['Quarter'] = df['Date'].dt.quarter
    
    # Calculate Year-over-Year price changes
    df['Price_YoY_Growth_Pct'] = df['Median_Price'].pct_change(periods=4) * 100
    
    # Flag extreme market spikes or drops using statistics
    price_mean = df['Median_Price'].mean()
    price_std = df['Median_Price'].std()
    df['Is_Outlier'] = np.where(np.abs((df['Median_Price'] - price_mean) / price_std) > 3, 1, 0)
    
    print("🎉 Data transformation complete.")
    return df

if __name__ == "__main__":
    raw_data = fetch_real_estate_data()
    if raw_data is not None:
        processed_data = clean_and_transform(raw_data)
        os.makedirs("data/processed", exist_ok=True)
        output_path = "data/processed/cleaned_us_housing_market.csv"
        processed_data.to_csv(output_path, index=False)
        print(f"💾 Cleaned dataset saved to: {output_path}")
