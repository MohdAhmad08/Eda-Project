# 1.  UNDERSTAND THE PROBLEM
# Dataset: US Border Crossing Entry Data
# Goal: Explore traffic patterns by date, border, state, and crossing type
# -----------------------------------

# 2.  IMPORT LIBRARIES
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Set style
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (14, 6)

# 3.  LOAD THE DATASET
# -----------------------------------
df = pd.read_csv("/Border_Crossing_Entry_Data.csv")

# 4.  DATA PREPROCESSING
# -----------------------------------

# Check for missing values
print("🔹 Missing values:\n", df.isnull().sum(), "\n")

# Drop duplicates
df = df.drop_duplicates()

# Convert 'Date' column to datetime
df["Date"] = pd.to_datetime(df["Date"], format="%b %Y", errors="coerce")

# Drop rows with missing or invalid dates
df = df.dropna(subset=["Date"])


df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month_name()
df["Month_Num"] = df["Date"].dt.month  # for ordering later


# 5.  INITIAL DATA EXPLORATION
# -----------------------------------

print("🔹 Head:\n", df.head(), "\n")
print("🔹 Tail:\n", df.tail(), "\n")
print("🔹 Info:")
df.info()
print("\n🔹 Summary Stats:\n", df.describe(include='all'), "\n")
print("🔹 Unique values per column:\n", df.nunique())

# 6.  UNIVARIATE ANALYSIS
# -----------------------------------

# Border count
sns.countplot(x="Border", data=df, palette="Set2")
plt.title("Count of Records by Border")
plt.show()

# Measure type distribution
sns.countplot(y="Measure", data=df, order=df["Measure"].value_counts().index, palette="coolwarm")
plt.title("Distribution of Measures (Crossing Types)")
plt.show()

# 7.  BIVARIATE ANALYSIS
# -----------------------------------

# Border vs Value
sns.boxplot(x="Border", y="Value", data=df, palette="Set3")
plt.title("Distribution of Crossing Values by Border")
plt.show()

# Date vs Value (trend)
monthly_trend = df.groupby("Date")["Value"].sum().reset_index()
sns.lineplot(x="Date", y="Value", data=monthly_trend)
plt.title("Monthly Total Border Crossings")
plt.xticks(rotation=45)
plt.show()


# 8.  MULTIVARIATE ANALYSIS
# -----------------------------------

# Top 10 states by total crossings
top_states = df.groupby("State")["Value"].sum().sort_values(ascending=False).head(10)
sns.barplot(x=top_states.values, y=top_states.index, palette="viridis")
plt.title("Top 10 States by Total Crossings")
plt.xlabel("Total Crossings")
plt.ylabel("State")
plt.show()

# Heatmap: Monthly Seasonality
# Group and pivot the data
pivot = df.pivot_table(index="Month", columns="Year", values="Value", aggfunc="sum")

# Ensure month order
month_order = ['January','February','March','April','May','June','July','August','September','October','November','December']
pivot = pivot.reindex(month_order)

# Plot a larger, clearer heatmap
plt.figure(figsize=(18, 10))
sns.heatmap(
    pivot,
    cmap="YlGnBu",
    linewidths=1,
    linecolor='white',
    annot=True,
    fmt=".0f",
    annot_kws={"size": 8},
    cbar_kws={"shrink": 0.7},
    square=True
)
plt.title("Seasonal Heatmap of Border Crossings (Monthly Totals)", fontsize=16)
plt.xlabel("Year", fontsize=12)
plt.ylabel("Month", fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()



# 9.  OUTLIER DETECTION
# -----------------------------------

sns.boxplot(x=df["Value"], color="orange")
plt.title("Outlier Detection in Crossing Values")
plt.show()


# 10. INSIGHT SUMMARY (Manual/Report)
# -----------------------------------
# Example: High crossing volumes are seen in states bordering Mexico, Trucks dominate Measure type,
# and crossings drop in winter months, peak in summer.

