import pandas as pd
import matplotlib.pyplot as plt

# Load the cleaned housing-market dataset
file_path = "data/processed/cleaned_us_housing_market.csv"

df = pd.read_csv(file_path)

# Convert Date column to datetime
df["Date"] = pd.to_datetime(df["Date"])

# Sort data chronologically
df = df.sort_values("Date").reset_index(drop=True)

# -----------------------------
# DATASET OVERVIEW
# -----------------------------

print("\n--- Dataset Shape ---")
print(df.shape)

print("\n--- Columns ---")
print(df.columns.tolist())

print("\n--- Date Range ---")
print("Start:", df["Date"].min())
print("End:", df["Date"].max())

print("\n--- First 5 Records ---")
print(df.head())

print("\n--- Last 5 Records ---")
print(df.tail())

# -----------------------------
# DATA QUALITY CHECK
# -----------------------------

print("\n--- Missing Values ---")
print(df.isnull().sum())

print("\n--- Duplicate Dates ---")
print(df["Date"].duplicated().sum())

# -----------------------------
# TIME INTERVAL CHECK
# -----------------------------

print("\n--- Date Differences ---")
print(df["Date"].diff().value_counts().head())

# -----------------------------
# TARGET VARIABLE
# -----------------------------

print("\n--- Median Price Statistics ---")
print(df["Median_Price"].describe())

# -----------------------------
# MEDIAN PRICE TIME-SERIES PLOT
# -----------------------------

plt.figure(figsize=(12, 6))

plt.plot(
    df["Date"],
    df["Median_Price"],
    linewidth=2
)

plt.title("U.S. Median House Price Over Time")
plt.xlabel("Year")
plt.ylabel("Median House Price ($)")

plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.show()


# -----------------------------
# QUARTERLY SEASONALITY ANALYSIS
# -----------------------------

quarterly_avg = df.groupby("Quarter")["Median_Price"].mean()

print("\n--- Average Median Price by Quarter ---")
print(quarterly_avg)

plt.figure(figsize=(8, 5))

plt.bar(
    quarterly_avg.index,
    quarterly_avg.values
)

plt.title("Average U.S. Median House Price by Quarter")
plt.xlabel("Quarter")
plt.ylabel("Average Median House Price ($)")
plt.xticks([1, 2, 3, 4], ["Q1", "Q2", "Q3", "Q4"])

plt.tight_layout()
plt.show()


# -----------------------------
# YOY GROWTH BY QUARTER
# -----------------------------

yoy_quarterly_avg = df.groupby("Quarter")["Price_YoY_Growth_Pct"].mean()

print("\n--- Average YoY Price Growth by Quarter ---")
print(yoy_quarterly_avg)

plt.figure(figsize=(8, 5))

plt.bar(
    yoy_quarterly_avg.index,
    yoy_quarterly_avg.values
)

plt.title("Average YoY Housing Price Growth by Quarter")
plt.xlabel("Quarter")
plt.ylabel("Average YoY Growth (%)")
plt.xticks([1, 2, 3, 4], ["Q1", "Q2", "Q3", "Q4"])

plt.axhline(0, linewidth=1)

plt.tight_layout()
plt.show()

# -----------------------------
# STATIONARITY TEST - ADF
# -----------------------------

from statsmodels.tsa.stattools import adfuller

# Remove missing values
price_series = df["Median_Price"].dropna()

# Perform Augmented Dickey-Fuller test
adf_result = adfuller(price_series)

print("\n--- Augmented Dickey-Fuller Test ---")
print("ADF Statistic:", adf_result[0])
print("p-value:", adf_result[1])

print("\nCritical Values:")
for key, value in adf_result[4].items():
    print(f"{key}: {value}")

# Interpretation
if adf_result[1] <= 0.05:
    print("\nResult: The series is likely STATIONARY.")
else:
    print("\nResult: The series is likely NON-STATIONARY.")


    # -----------------------------
# FIRST DIFFERENCING
# -----------------------------

# Calculate first difference of Median Price
price_diff = price_series.diff().dropna()

print("\n--- First Differenced Median Price ---")
print(price_diff.head())

# ADF test on differenced series
adf_diff_result = adfuller(price_diff)

print("\n--- ADF Test After First Differencing ---")
print("ADF Statistic:", adf_diff_result[0])
print("p-value:", adf_diff_result[1])

print("\nCritical Values:")
for key, value in adf_diff_result[4].items():
    print(f"{key}: {value}")

if adf_diff_result[1] <= 0.05:
    print("\nResult: The differenced series is STATIONARY.")
else:
    print("\nResult: The differenced series is still NON-STATIONARY.")

    # -----------------------------
# AUTOCORRELATION ANALYSIS
# -----------------------------

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Plot ACF of the stationary series
plt.figure(figsize=(10, 5))
plot_acf(price_diff, lags=20, ax=plt.gca())

plt.title("ACF of First-Differenced Median House Price")
plt.xlabel("Lag")
plt.ylabel("Autocorrelation")

plt.tight_layout()
plt.show()


# Plot PACF of the stationary series
plt.figure(figsize=(10, 5))
plot_pacf(price_diff, lags=20, ax=plt.gca(), method="ywm")

plt.title("PACF of First-Differenced Median House Price")
plt.xlabel("Lag")
plt.ylabel("Partial Autocorrelation")

plt.tight_layout()
plt.show()