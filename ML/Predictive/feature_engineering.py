import pandas as pd
from pathlib import Path


# ---------------------------------------------------------
# 1. Locate the project root and input dataset
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

input_file = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cleaned_us_housing_market.csv"
)

output_file = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ml_features.csv"
)


# ---------------------------------------------------------
# 2. Load the cleaned housing dataset
# ---------------------------------------------------------

df = pd.read_csv(input_file)

df["Date"] = pd.to_datetime(df["Date"])

df = df.sort_values("Date").reset_index(drop=True)


print("\n--- Feature Engineering: Data Loaded ---")
print("Dataset shape:", df.shape)
print("Date range:", df["Date"].min(), "to", df["Date"].max())


# ---------------------------------------------------------
# 3. Create a chronological time index
# ---------------------------------------------------------

df["Time_Index"] = range(len(df))


# ---------------------------------------------------------
# 4. Create lag features
# ---------------------------------------------------------

# Previous quarter's house price
df["Price_Lag_1"] = df["Median_Price"].shift(1)

# Price from the same quarter one year earlier
df["Price_Lag_4"] = df["Median_Price"].shift(4)


# ---------------------------------------------------------
# 5. Create a rolling historical price feature
# ---------------------------------------------------------

# Average price of the previous four quarters.
# shift(1) ensures the current quarter is NOT included.
df["Rolling_Mean_4"] = (
    df["Median_Price"]
    .shift(1)
    .rolling(window=4)
    .mean()
)


# ---------------------------------------------------------
# 6. Select ML features
# ---------------------------------------------------------

feature_columns = [
    "Year",
    "Quarter",
    "Time_Index",
    "Price_Lag_1",
    "Price_Lag_4",
    "Rolling_Mean_4"
]

target_column = "Median_Price"


ml_df = df[
    ["Date"] + feature_columns + [target_column]
].copy()


# ---------------------------------------------------------
# 7. Remove rows where historical lag information
#    is not available
# ---------------------------------------------------------

ml_df = ml_df.dropna().reset_index(drop=True)


# ---------------------------------------------------------
# 8. Save the ML-ready dataset
# ---------------------------------------------------------

output_file.parent.mkdir(
    parents=True,
    exist_ok=True
)

ml_df.to_csv(output_file, index=False)


# ---------------------------------------------------------
# 9. Display the result
# ---------------------------------------------------------

print("\n--- ML Feature Dataset ---")
print("Shape:", ml_df.shape)

print("\n--- Columns ---")
print(ml_df.columns.tolist())

print("\n--- First 5 Records ---")
print(ml_df.head())

print("\n--- Missing Values ---")
print(ml_df.isnull().sum())

print("\nML feature dataset saved to:")
print(output_file)