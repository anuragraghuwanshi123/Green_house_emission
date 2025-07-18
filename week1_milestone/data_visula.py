import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load the CSV
file_path = r"C:\Users\HP\OneDrive\Desktop\greenhouse emission\week1_milestone\SupplyChainEmissionFactorsforUSIndustriesCommodities2015_Summary (2).csv"

# Check if file exists
if not os.path.exists(file_path):
    print("❌ File not found. Check the path.")
    exit()

# Read and clean
df = pd.read_csv(file_path)
df.columns = df.columns.str.strip()

# ─────────────────────────────────────────────────────────────
# 1️⃣ BAR PLOT: Top 10 Industries by Average CH₄ Emissions
# ─────────────────────────────────────────────────────────────
ch4_df = df[df['Substance'].str.upper().str.contains("CH4")]

ch4_by_industry = (
    ch4_df.groupby('Industry Name')['Supply Chain Emission Factors with Margins']
    .mean()
    .sort_values(ascending=False)
    .head(10)
)

if not ch4_by_industry.empty:
    plt.figure(figsize=(12, 6))
    ch4_by_industry.plot(kind='bar', color='mediumseagreen', edgecolor='black')
    plt.title("Top 10 Industries by Average CH₄ Emissions", fontsize=14)
    plt.ylabel("CH₄ Emission Factor (with Margins)")
    plt.xlabel("Industry")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(file_path), "ch4_bar_plot.png"))
    plt.show()
else:
    print("⚠️ No CH₄ emission data found.")

# ─────────────────────────────────────────────────────────────
# 2️⃣ HEATMAP: Correlation Between Gases
# ─────────────────────────────────────────────────────────────
pivot_df = df.pivot_table(
    index='Industry Name',
    columns='Substance',
    values='Supply Chain Emission Factors with Margins',
    aggfunc='mean'
)

pivot_df = pivot_df.dropna(axis=1, how='all')
corr_matrix = pivot_df.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='YlGnBu', fmt=".2f")
plt.title("Correlation Between Emission Gases", fontsize=14)
plt.tight_layout()
plt.show()

# ─────────────────────────────────────────────────────────────
# 3️⃣ BAR PLOT: Top 5 Emission Gases by Average Emission
# ─────────────────────────────────────────────────────────────
gas_avg = (
    df.groupby('Substance')['Supply Chain Emission Factors with Margins']
    .mean()
    .sort_values(ascending=False)
    .head(5)
)

plt.figure(figsize=(10, 5))
gas_avg.plot(kind='bar', color='salmon', edgecolor='black')
plt.title("Top 5 Emission Gases by Average Supply Chain Emissions", fontsize=14)
plt.ylabel("Average Emission Factor (with Margins)")
plt.xlabel("Gas (Substance)")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(file_path), "top5_gases_bar_plot.png"))
plt.show()



















