# =============================================================================
# WEEKS 2–3 – Dataset Validation, EDA, and Mortgage Rate Enrichment
# IDX Exchange MLS Analytics Program
# =============================================================================
# This script:
#   1. Inspects the combined datasets (structure, types, missing values)
#   2. Runs basic EDA on key numeric fields
#   3. Fetches the 30-year fixed mortgage rate from FRED
#   4. Merges mortgage rates onto both datasets by year-month
#
# Input:  data/output/sold_combined.csv
#         data/output/listings_combined.csv
# Output: data/output/sold_enriched.csv
#         data/output/listings_enriched.csv
#         data/output/sold_missing_report.csv
#         data/output/listings_missing_report.csv
# =============================================================================

import pandas as pd
import numpy as np
import os

DATA_DIR = "data/output"


sold = pd.read_csv(os.path.join(DATA_DIR, "sold_combined.csv"), low_memory=False)
listings = pd.read_csv(os.path.join(DATA_DIR, "listings_combined.csv"), low_memory=False)

print(f"SOLD:     {sold.shape[0]:,} rows × {sold.shape[1]} columns")
print(f"LISTINGS: {listings.shape[0]:,} rows × {listings.shape[1]} columns")


print("\n--- SOLD column data types ---")
print(sold.dtypes.value_counts())

print("\n--- LISTINGS column data types ---")
print(listings.dtypes.value_counts())

print("\n--- SOLD preview ---")
print(sold.head(3))


def missing_report(df, label):
    """
    Returns a DataFrame summarizing missing value counts and percentages
    for every column. Flags columns with >90% missing.
    """
    total = len(df)
    missing_counts = df.isnull().sum()
    missing_pct = (missing_counts / total * 100).round(2)

    report = pd.DataFrame({
        "missing_count": missing_counts,
        "missing_pct": missing_pct
    })
    report = report[report["missing_count"] > 0].sort_values("missing_pct", ascending=False)
    report["flag_90pct"] = report["missing_pct"] > 90

    print(f"\n--- {label} — Missing Value Report ---")
    print(f"Total columns with any missing: {len(report)}")
    print(f"Columns with >90% missing:      {report['flag_90pct'].sum()}")
    print(report.head(20).to_string())

    return report

sold_missing = missing_report(sold, "SOLD")
listings_missing = missing_report(listings, "LISTINGS")

sold_missing.to_csv(os.path.join(DATA_DIR, "sold_missing_report.csv"))
listings_missing.to_csv(os.path.join(DATA_DIR, "listings_missing_report.csv"))


NUMERIC_FIELDS = [
    "ClosePrice", "ListPrice", "OriginalListPrice",
    "LivingArea", "LotSizeAcres",
    "BedroomsTotal", "BathroomsTotalInteger",
    "DaysOnMarket", "YearBuilt"
]

sold_numeric_fields = [f for f in NUMERIC_FIELDS if f in sold.columns]

print("\n--- SOLD — Numeric Field Distribution Summary ---")
print(sold[sold_numeric_fields].describe(percentiles=[.05, .25, .50, .75, .95]).T.to_string())

# Suggested EDA questions
print("\n--- EDA Checks ---")

# 1. Median and average close price
if "ClosePrice" in sold.columns:
    print(f"Median ClosePrice:  ${sold['ClosePrice'].median():,.0f}")
    print(f"Average ClosePrice: ${sold['ClosePrice'].mean():,.0f}")

# 2. Days on market distribution
if "DaysOnMarket" in sold.columns:
    print(f"\nDaysOnMarket — median: {sold['DaysOnMarket'].median()}, "
          f"mean: {sold['DaysOnMarket'].mean():.1f}, "
          f"max: {sold['DaysOnMarket'].max()}")

# 3. Sold above vs. below list price
if "ClosePrice" in sold.columns and "ListPrice" in sold.columns:
    above = (sold["ClosePrice"] > sold["ListPrice"]).sum()
    below = (sold["ClosePrice"] < sold["ListPrice"]).sum()
    total_valid = sold[["ClosePrice", "ListPrice"]].dropna().shape[0]
    print(f"\nSold above list:  {above:,} ({above/total_valid*100:.1f}%)")
    print(f"Sold below list:  {below:,} ({below/total_valid*100:.1f}%)")

# 4. Date consistency check — CloseDate before ListingContractDate
date_fields = ["CloseDate", "ListingContractDate"]
if all(f in sold.columns for f in date_fields):
    sold_dates = sold[date_fields].dropna()
    sold_dates = sold_dates.apply(pd.to_datetime, errors="coerce")
    bad_dates = (sold_dates["CloseDate"] < sold_dates["ListingContractDate"]).sum()
    print(f"\nRecords where CloseDate < ListingContractDate: {bad_dates:,}")

# 5. County price breakdown
if "CountyOrParish" in sold.columns and "ClosePrice" in sold.columns:
    print("\nMedian ClosePrice by County (top 10):")
    county_prices = (
        sold.groupby("CountyOrParish")["ClosePrice"]
        .median()
        .sort_values(ascending=False)
        .head(10)
    )
    print(county_prices.apply(lambda x: f"${x:,.0f}").to_string())


print("\n--- Fetching FRED mortgage rate data ---")

fred_url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=MORTGAGE30US"

try:
    mortgage = pd.read_csv(fred_url, parse_dates=["observation_date"])
    mortgage.columns = ["date", "rate_30yr_fixed"]

    mortgage = mortgage.dropna(subset=["rate_30yr_fixed"])

    print(f"Fetched {len(mortgage):,} weekly rate observations")
    print(f"Date range: {mortgage['date'].min().date()} to {mortgage['date'].max().date()}")

except Exception as e:
    print(f"❌ Failed to fetch FRED data: {e}")
    raise

mortgage["year_month"] = mortgage["date"].dt.to_period("M")

mortgage_monthly = (
    mortgage
    .groupby("year_month")["rate_30yr_fixed"]
    .mean()
    .reset_index()
)

print(f"\nMonthly rate observations: {len(mortgage_monthly)}")
print(mortgage_monthly.tail(5).to_string(index=False))

sold["CloseDate"] = pd.to_datetime(sold["CloseDate"], errors="coerce")
sold["year_month"] = sold["CloseDate"].dt.to_period("M")


listings["ListingContractDate"] = pd.to_datetime(listings["ListingContractDate"], errors="coerce")
listings["year_month"] = listings["ListingContractDate"].dt.to_period("M")


sold_enriched = sold.merge(mortgage_monthly, on="year_month", how="left")
listings_enriched = listings.merge(mortgage_monthly, on="year_month", how="left")


sold_null_rate = sold_enriched["rate_30yr_fixed"].isnull().sum()
listings_null_rate = listings_enriched["rate_30yr_fixed"].isnull().sum()

print(f"\nSOLD — rows with null mortgage rate after merge:     {sold_null_rate:,}")
print(f"LISTINGS — rows with null mortgage rate after merge: {listings_null_rate:,}")

if sold_null_rate > 0:
    print("  ⚠️  Unmatched sold months:")
    print(sold_enriched[sold_enriched["rate_30yr_fixed"].isnull()]["year_month"].value_counts())

if listings_null_rate > 0:
    print("  ⚠️  Unmatched listing months:")
    print(listings_enriched[listings_enriched["rate_30yr_fixed"].isnull()]["year_month"].value_counts())


print("\n--- Merge preview (SOLD) ---")
preview_cols = ["CloseDate", "year_month", "ClosePrice", "rate_30yr_fixed"]
preview_cols = [c for c in preview_cols if c in sold_enriched.columns]
print(sold_enriched[preview_cols].dropna().head(5).to_string(index=False))


sold_enriched["year_month"] = sold_enriched["year_month"].astype(str)
listings_enriched["year_month"] = listings_enriched["year_month"].astype(str)

sold_enriched.to_csv(os.path.join(DATA_DIR, "sold_enriched.csv"), index=False)
listings_enriched.to_csv(os.path.join(DATA_DIR, "listings_enriched.csv"), index=False)

print(f"\n Saved: data/output/sold_enriched.csv ({len(sold_enriched):,} rows)")
print(f" Saved: data/output/listings_enriched.csv ({len(listings_enriched):,} rows)")
print(f" Saved: data/output/sold_missing_report.csv")
print(f" Saved: data/output/listings_missing_report.csv")