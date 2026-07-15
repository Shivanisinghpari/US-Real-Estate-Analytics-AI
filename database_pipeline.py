import os
import sqlite3
import pandas as pd

def initialize_database(db_path):
    """Creates a local SQLite database and establishes the analytics table schema."""
    print("🗄️ Initializing SQLite database...")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create a structured SQL table for the housing data
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS housing_market_trends (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            record_date TEXT NOT NULL,
            median_price REAL NOT NULL,
            year INTEGER,
            quarter INTEGER,
            price_yoy_growth_pct REAL,
            is_outlier INTEGER
        );
    """)
    conn.commit()
    print("✅ SQL Table 'housing_market_trends' verified/created.")
    return conn

def load_csv_to_sql(csv_path, conn):
    """Reads the processed CSV file and transfers records into the SQL database."""
    print("⏳ Ingesting cleaned data pipeline into SQL database...")
    try:
        # Read the cleaned CSV file from Step 2
        df = pd.read_csv(csv_path)
        
        # Map CSV columns to match the SQL table columns
        df_sql = df.rename(columns={
            'Date': 'record_date',
            'Median_Price': 'median_price',
            'Year': 'year',
            'Quarter': 'quarter',
            'Price_YoY_Growth_Pct': 'price_yoy_growth_pct',
            'Is_Outlier': 'is_outlier'
        })
        
        # Insert data into SQLite table (replace if data already exists to avoid duplicates)
        df_sql.to_sql('housing_market_trends', conn, if_exists='replace', index=False)
        
        # Verify ingestion by running a quick count query
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM housing_market_trends;")
        row_count = cursor.fetchone()[0]
        
        print(f"🚀 Successfully migrated {row_count} historical records into the database.")
    except Exception as e:
        print(f"❌ Database ingestion failed: {e}")

if __name__ == "__main__":
    cleaned_data_source = "data/processed/cleaned_us_housing_market.csv"
    database_destination = "data/processed/real_estate_analytics.db"
    
    # Run the database pipeline
    if os.path.exists(cleaned_data_source):
        db_connection = initialize_database(database_destination)
        load_csv_to_sql(cleaned_data_source, db_connection)
        db_connection.close()
        print("💾 Database transaction pipeline complete. Connection closed.")
    else:
        print(f"❌ Source file not found at {cleaned_data_source}. Please run data_pipeline.py first.")

