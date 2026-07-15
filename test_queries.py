import sqlite3
import pandas as pd

# Connect to the database created by your pipeline
db_path = "data/processed/real_estate_analytics.db"
conn = sqlite3.connect(db_path)

try:
    print("📋 Fetching top 5 rows from SQL database...")
    
    # 1. Standard SQL query string
    query = """
        SELECT record_date, median_price, price_yoy_growth_pct 
        FROM housing_market_trends 
        WHERE is_outlier = 0
        LIMIT 5;
    """
    
    # 2. Use Pandas to run the query and print a beautiful table
    df_results = pd.read_sql_query(query, conn)
    
    print("\n--- SQL QUERY RESULT ---")
    print(df_results)
    print("------------------------\n")
    print("✅ SQL query executed perfectly with zero errors!")

except Exception as e:
    # 3. Catch and display errors clearly
    print(f"❌ SQL Error encountered: {e}")

finally:
    conn.close()
