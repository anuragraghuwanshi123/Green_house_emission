import pandas as pd
import matplotlib.pyplot as plt
import os

# Load the CSV
file_path = r"C:\Users\HP\OneDrive\Desktop\greenhouse emission\week1_milestone\SupplyChainEmissionFactorsforUSIndustriesCommodities2015_Summary (2).csv"

# Check file exists
if not os.path.exists(file_path):
    print("❌ File not found. Check the path.")
    exit()

# Read and clean
df = pd.read_csv(file_path)
df.columns = df.columns.str.strip()  # Remove leading/trailing spaces from column names

# Drop rows with missing values only in key columns
df = df.dropna(subset=['Industry Name', 'Supply Chain Emission Factors with Margins'])

# ─────────────────────────────────────────────────────────────
# 1️⃣ BAR PLOT: Top 10 Industries with Highest Total Emissions
# ─────────────────────────────────────────────────────────────
industry_total_emissions = (
    df.groupby('Industry Name')['Supply Chain Emission Factors with Margins']
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

if not industry_total_emissions.empty:
    plt.figure(figsize=(12, 6))
    industry_total_emissions.plot(kind='bar', color='darkorange', edgecolor='black')
    plt.title("Top 10 Industries with Highest Total Emissions", fontsize=14)
    plt.ylabel("Total Emission Factor (with Margins)")
    plt.xlabel("Industry")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(file_path), "top10_industries_bar.png"))
    plt.show()
else:
    print("⚠️ No data available for industry emissions.")

# ─────────────────────────────────────────────────────────────
# 2️⃣ HISTOGRAM: Distribution of Emission Values
# ─────────────────────────────────────────────────────────────
if not df['Supply Chain Emission Factors with Margins'].empty:
    plt.figure(figsize=(10, 5))
    plt.hist(df['Supply Chain Emission Factors with Margins'], bins=30, color='steelblue', edgecolor='black')
    plt.title("Distribution of Emission Values", fontsize=14)
    plt.xlabel("Emission Factor (with Margins)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(file_path), "emission_distribution_hist.png"))
    plt.show()
else:
    print("⚠️ No emission data available to plot histogram.")

