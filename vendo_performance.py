import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
import warnings
from scipy import stats
from scipy.stats import ttest_ind, t

# Ignore warning messages
warnings.filterwarnings("ignore")

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def format_dollars(value):
    """
    Convert large dollar values into readable format.
    Examples:
        1000 -> $1.00K
        1000000 -> $1.00M
    """
    if pd.isna(value):
        return "$0.00"
    if value >= 1_000_000:
        return f"${value / 1_000_000:.2f}M"
    elif value >= 1_000:
        return f"${value / 1_000:.2f}K"
    else:
        return f"${value:.2f}"


def calculate_confidence_interval(data, confidence=0.95):
    """
    Calculate confidence interval for a pandas Series or array-like object.
    Returns (lower_bound, upper_bound).
    """
    data = pd.Series(data).dropna()
    n = len(data)
    if n < 2:
        return (np.nan, np.nan)

    mean = np.mean(data)
    std_error = stats.sem(data)
    margin = std_error * t.ppf((1 + confidence) / 2, n - 1)
    return mean - margin, mean + margin


def safe_qcut(series, q=3, labels=None):
    """
    Safe wrapper around pd.qcut that drops duplicate bin edges.
    """
    return pd.qcut(series, q=q, labels=labels, duplicates="drop")


# ============================================================
# DATABASE CONNECTION
# ============================================================

conn = sqlite3.connect("inventory.db")

# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_sql_query(
    """
    SELECT *
    FROM vendor_sales_summary
    WHERE GrossProfit > 0
      AND ProfitMargin > 0
      AND TotalSalesQuantity > 0
    """,
    conn,
)

print("\nFirst 5 Rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns)

# ============================================================
# DESCRIPTIVE STATISTICS
# ============================================================

print("\nDescriptive Statistics:")
print(df.describe(include="all"))

# ============================================================
# HISTOGRAM + BOXPLOT FOR NUMERICAL COLUMNS
# ============================================================

numerical_cols = df.select_dtypes(include=[np.number]).columns

for col in numerical_cols:
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    sns.histplot(df[col], kde=True)
    plt.title(f"Histogram of {col}")

    plt.subplot(1, 2, 2)
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot of {col}")

    plt.tight_layout()
    plt.show()

# ============================================================
# CORRELATION HEATMAP
# ============================================================

correlation_matrix = df[numerical_cols].corr()

plt.figure(figsize=(12, 8))
sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    linewidths=0.5,
    cbar=True,
)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

# ============================================================
# BRAND PERFORMANCE ANALYSIS
# ============================================================

brand_performance = df.groupby("Description").agg(
    TotalSalesDollars=("TotalSalesDollars", "sum"),
    ProfitMargin=("ProfitMargin", "mean"),
).reset_index()

low_sales_threshold = brand_performance["TotalSalesDollars"].quantile(0.15)
high_margin_threshold = brand_performance["ProfitMargin"].quantile(0.85)

print("\nLow Sales Threshold:")
print(low_sales_threshold)

print("\nHigh Margin Threshold:")
print(high_margin_threshold)

target_brands = brand_performance[
    (brand_performance["TotalSalesDollars"] < low_sales_threshold)
    & (brand_performance["ProfitMargin"] > high_margin_threshold)
]

print("\nTarget Brands for Intervention:")
print(target_brands)

plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=brand_performance,
    x="TotalSalesDollars",
    y="ProfitMargin",
    hue="Description",
    palette="tab10",
    legend=False,
)
plt.title("Brand Performance: Sales vs Profit Margin")
plt.xlabel("Total Sales Dollars")
plt.ylabel("Profit Margin")
plt.axvline(
    x=low_sales_threshold,
    color="red",
    linestyle="--",
    label="Low Sales Threshold",
)
plt.axhline(
    y=high_margin_threshold,
    color="blue",
    linestyle="--",
    label="High Margin Threshold",
)
plt.legend()
plt.tight_layout()
plt.show()

# ============================================================
# TOP 10 VENDORS AND BRANDS
# ============================================================

top_vendors_sales = (
    df.groupby("VendorName")["TotalSalesDollars"]
    .sum()
    .nlargest(10)
    .reset_index()
)

top_brands_sales = (
    df.groupby("Description")["TotalSalesDollars"]
    .sum()
    .nlargest(10)
    .reset_index()
)

print("\nTop Vendors:")
print(top_vendors_sales)

print("\nTop Brands:")
print(top_brands_sales)

plt.figure(figsize=(16, 6))

ax1 = plt.subplot(1, 2, 1)
sns.barplot(
    y=top_vendors_sales["VendorName"],
    x=top_vendors_sales["TotalSalesDollars"],
    palette="Blues_r",
    ax=ax1,
)
ax1.set_title("Top 10 Vendors by Sales")
for bar in ax1.patches:
    ax1.text(
        bar.get_width() + bar.get_width() * 0.02,
        bar.get_y() + bar.get_height() / 2,
        format_dollars(bar.get_width()),
        ha="left",
        va="center",
        fontsize=10,
        color="black",
    )
ax1.set_xlabel("")
ax1.set_ylabel("")

ax2 = plt.subplot(1, 2, 2)
sns.barplot(
    y=top_brands_sales["Description"].astype(str),
    x=top_brands_sales["TotalSalesDollars"],
    palette="Reds_r",
    ax=ax2,
)
ax2.set_title("Top 10 Brands by Sales")
for bar in ax2.patches:
    ax2.text(
        bar.get_width() + bar.get_width() * 0.02,
        bar.get_y() + bar.get_height() / 2,
        format_dollars(bar.get_width()),
        ha="left",
        va="center",
        fontsize=10,
        color="black",
    )
ax2.set_xlabel("")
ax2.set_ylabel("")
plt.tight_layout()
plt.show()

# ============================================================
# VENDOR PERFORMANCE ANALYSIS
# ============================================================

vendor_performance = df.groupby("VendorName").agg(
    TotalPurchaseDollars=("TotalPurchaseDollars", "sum"),
    TotalSalesDollars=("TotalSalesDollars", "sum"),
    GrossProfit=("GrossProfit", "sum"),
).reset_index()

vendor_performance["PurchaseContribution%"] = (
    vendor_performance["TotalPurchaseDollars"]
    / vendor_performance["TotalPurchaseDollars"].sum()
) * 100

vendor_performance = vendor_performance.sort_values(
    by="PurchaseContribution%",
    ascending=False,
).copy()

top_vendors = vendor_performance.head(10).copy()
top_vendors["CumulativeContribution%"] = top_vendors["PurchaseContribution%"].cumsum()

print("\nVendor Performance:")
print(top_vendors)

# ============================================================
# PARETO CHART - VENDOR CONTRIBUTION
# ============================================================

fig, ax1 = plt.subplots(figsize=(12, 6))

sns.barplot(
    x=top_vendors["VendorName"],
    y=top_vendors["PurchaseContribution%"],
    palette="mako",
    ax=ax1,
)

for i, value in enumerate(top_vendors["PurchaseContribution%"]):
    ax1.text(
        i,
        value - 1,
        f"{value:.2f}%",
        ha="center",
        fontsize=9,
        color="white",
    )

ax2 = ax1.twinx()
ax2.plot(
    top_vendors["VendorName"],
    top_vendors["CumulativeContribution%"],
    color="red",
    marker="o",
)

ax1.set_xticklabels(top_vendors["VendorName"], rotation=90)
ax1.set_ylabel("Purchase Contribution %")
ax2.set_ylabel("Cumulative Contribution %")
ax1.set_xlabel("Vendors")
ax1.set_title("Pareto Chart: Vendor Contribution to Total Purchases")
ax2.axhline(y=100, color="gray", linestyle="dashed")
plt.tight_layout()
plt.show()

# ============================================================
# TOTAL PURCHASE CONTRIBUTION OF TOP 10 VENDORS
# ============================================================

total_top10_contribution = round(top_vendors["PurchaseContribution%"].sum(), 2)
print(f"\nTotal Purchase Contribution of Top 10 Vendors is {total_top10_contribution}%")

# ============================================================
# DONUT PIE CHART - TOP VENDOR CONTRIBUTION
# ============================================================

plt.figure(figsize=(10, 10))

plt.pie(
    top_vendors["PurchaseContribution%"],
    labels=top_vendors["VendorName"],
    autopct="%1.1f%%",
    startangle=90,
    pctdistance=0.85,
)

centre_circle = plt.Circle((0, 0), 0.70, fc="white")
fig = plt.gcf()
fig.gca().add_artist(centre_circle)

plt.text(
    0,
    0,
    f"Top 10 Total:\n{total_top10_contribution}%",
    ha="center",
    va="center",
    fontsize=16,
    fontweight="bold",
)

plt.title("Vendor Contribution to Total Procurement")
plt.tight_layout()
plt.show()

# ============================================================
# BULK PURCHASING ANALYSIS
# ============================================================

df["UnitPurchasePrice"] = df["TotalPurchaseDollars"] / df["TotalPurchaseQuantity"]

df["OrderSize"] = safe_qcut(
    df["TotalPurchaseQuantity"],
    q=3,
    labels=["Small", "Medium", "Large"],
)

bulk_purchase_analysis = df.groupby("OrderSize", observed=False)["UnitPurchasePrice"].mean()

print("\nAverage Unit Purchase Price by Order Size:")
print(bulk_purchase_analysis)

plt.figure(figsize=(10, 6))
sns.boxplot(
    data=df,
    x="OrderSize",
    y="UnitPurchasePrice",
    palette="Set2",
)
plt.title("Impact of Bulk Purchasing on Unit Price")
plt.xlabel("Order Size")
plt.ylabel("Average Unit Purchase Price")
plt.tight_layout()
plt.show()

# ============================================================
# INVENTORY TURNOVER ANALYSIS
# ============================================================

low_inventory_turnover = df.groupby("VendorName").agg(
    StockTurnover=("StockTurnover", "mean"),
    TotalSalesDollars=("TotalSalesDollars", "sum"),
).reset_index()

low_inventory_turnover = low_inventory_turnover.sort_values(
    by="StockTurnover",
    ascending=True,
).head(10)

print("\nVendors with Low Inventory Turnover:")
print(low_inventory_turnover)

plt.figure(figsize=(12, 6))
sns.barplot(
    data=low_inventory_turnover,
    x="StockTurnover",
    y="VendorName",
    palette="rocket",
)
plt.title("Vendors with Low Inventory Turnover (Slow Moving Inventory)")
plt.xlabel("Average Stock Turnover")
plt.ylabel("Vendor Name")
plt.tight_layout()
plt.show()

# ============================================================
# UNSOLD INVENTORY ANALYSIS
# ============================================================

df["UnsoldInventoryValue"] = (
    (df["TotalPurchaseQuantity"] - df["TotalSalesQuantity"])
    * df["PurchasePrice"]
)

total_unsold_capital = df["UnsoldInventoryValue"].sum()
print("\nTotal Unsold Capital:", format_dollars(total_unsold_capital))

unsold_inventory_vendor = df.groupby("VendorName").agg(
    UnsoldInventoryValue=("UnsoldInventoryValue", "sum")
).reset_index()

unsold_inventory_vendor = unsold_inventory_vendor.sort_values(
    by="UnsoldInventoryValue",
    ascending=False,
).head(10)

print("\nTop Vendors by Unsold Inventory Value:")
print(unsold_inventory_vendor)

plt.figure(figsize=(12, 6))
sns.barplot(
    data=unsold_inventory_vendor,
    y="VendorName",
    x="UnsoldInventoryValue",
    palette="flare",
)
plt.title("Top Vendors Contributing to Unsold Inventory Capital")
plt.xlabel("Unsold Inventory Value")
plt.ylabel("Vendor Name")
plt.tight_layout()
plt.show()

# ============================================================
# CONFIDENCE INTERVAL ANALYSIS FOR PROFIT MARGINS
# ============================================================

vendor_profit_analysis = df.groupby("VendorName").agg(
    ProfitMargin=("ProfitMargin", "mean"),
    TotalSalesDollars=("TotalSalesDollars", "sum"),
).reset_index()

vendor_profit_analysis = vendor_profit_analysis.sort_values(
    by="TotalSalesDollars",
    ascending=False,
)

top_performing_vendors = vendor_profit_analysis.head(10).copy()
low_performing_vendors = vendor_profit_analysis.tail(10).copy()

top_ci = calculate_confidence_interval(top_performing_vendors["ProfitMargin"])
low_ci = calculate_confidence_interval(low_performing_vendors["ProfitMargin"])

print("\n95% Confidence Interval for Top-Performing Vendors:")
print(top_ci)

print("\n95% Confidence Interval for Low-Performing Vendors:")
print(low_ci)

plt.figure(figsize=(10, 5))
sns.histplot(top_performing_vendors["ProfitMargin"], bins=10, kde=True)
plt.title("Histogram of Profit Margins - Top Vendors")
plt.xlabel("Profit Margin")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))
sns.histplot(low_performing_vendors["ProfitMargin"], bins=10, kde=True, color="red")
plt.title("Histogram of Profit Margins - Low Vendors")
plt.xlabel("Profit Margin")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
categories = ["Top Vendors", "Low Vendors"]
means = [
    top_performing_vendors["ProfitMargin"].mean(),
    low_performing_vendors["ProfitMargin"].mean(),
]
errors = [
    (top_ci[1] - top_ci[0]) / 2 if pd.notna(top_ci[0]) else 0,
    (low_ci[1] - low_ci[0]) / 2 if pd.notna(low_ci[0]) else 0,
]

plt.errorbar(
    categories,
    means,
    yerr=errors,
    fmt="o",
    capsize=5,
)
plt.title("95% Confidence Intervals for Vendor Profit Margins")
plt.ylabel("Profit Margin")
plt.tight_layout()
plt.show()

# ============================================================
# HYPOTHESIS TESTING: U0 AND U1 WITH T-TEST
# ============================================================

# Null Hypothesis (U0):
# The mean profit margin of top-performing vendors is equal to the mean profit margin of low-performing vendors.
#
# Alternative Hypothesis (U1):
# The mean profit margin of top-performing vendors is different from the mean profit margin of low-performing vendors.

top_profit = top_performing_vendors["ProfitMargin"].dropna()
low_profit = low_performing_vendors["ProfitMargin"].dropna()

# Independent two-sample t-test
t_stat, p_value = ttest_ind(top_profit, low_profit, equal_var=False)

print("\nHypothesis Test: Top vs Low Performing Vendors")
print("U0: Mean profit margins are equal")
print("U1: Mean profit margins are different")
print(f"t-statistic: {t_stat:.4f}")
print(f"p-value: {p_value:.6f}")

alpha = 0.05
if p_value < alpha:
    print("Result: Reject U0. There is a significant difference in profit margins.")
else:
    print("Result: Fail to reject U0. No significant difference detected.")

# Optional: compare histograms side by side
plt.figure(figsize=(10, 5))
sns.histplot(top_profit, bins=10, kde=True, label="Top Vendors", stat="density", element="step", fill=False)
sns.histplot(low_profit, bins=10, kde=True, label="Low Vendors", stat="density", element="step", fill=False)
plt.title("Profit Margin Distribution: Top vs Low Vendors")
plt.xlabel("Profit Margin")
plt.ylabel("Density")
plt.legend()
plt.tight_layout()
plt.show()

# ============================================================
# CLOSE DATABASE CONNECTION
# ============================================================

conn.close()

print("\nAnalysis Completed Successfully!")
