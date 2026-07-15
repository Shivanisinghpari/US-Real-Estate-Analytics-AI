import os
import sqlite3
import pandas as pd

def initialize_database(db_path):
    # Initialize connection and set table schema
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
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
    print("LOG: Database initialized. Table verified.")
    return conn

def load_csv_to_sql(csv_path, conn):
    print("LOG: Starting data migration to SQL...")
    try:
        df = pd.read_csv(csv_path)
        
        # Rename columns to match database schema
        df_sql = df.rename(columns={
            'Date': 'record_date',
            'Median_Price': 'median_price',
            'Year': 'year',
            'Quarter': 'quarter',
            'Price_YoY_Growth_Pct': 'price_yoy_growth_pct',
            'Is_Outlier': 'is_outlier'
        })
        
        # Insert records into SQLite table
        df_sql.to_sql('housing_market_trends', conn, if_exists='replace', index=False)
        
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM housing_market_trends;")
        row_count = cursor.fetchone()[0]
        print(f"SUCCESS: Ingested {row_count} records into SQL table.")
        
    except Exception as e:
        print(f"ERROR: Ingestion failed. Details: {e}")

if __name__ == "__main__":
    src_file = "data/processed/cleaned_us_housing_market.csv"
    db_file = "data/processed/real_estate_analytics.db"
    
    if os.path.exists(src_file):
        db_conn = initialize_database(db_file)
        load_csv_to_sql(src_file, db_conn)
        db_conn.close()
        print("LOG: Pipeline finished execution successfully.")
    else:
        print(f"ERROR: Missing source file at {src_file}")
