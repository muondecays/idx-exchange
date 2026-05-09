# =============================================================================
# WEEK 6 – Feature Engineering and Market Metrics
# IDX Exchange MLS Analytics Program
# =============================================================================
# Engineers key housing market metrics from the cleaned datasets and produces
# segmented summary tables by property type and county.
#
# Input:  data/output/sold_cleaned.csv
#         data/output/listings_cleaned.csv
# Output: data/output/sold_features.csv
#         data/output/listings_features.csv
#         data/output/summary_by_property_type.csv
#         data/output/summary_by_county.csv
# =============================================================================

import pandas as pd
import numpy as np
import os

DATA_DIR = "data/output"

sold = pd.read_csv(os.path.join(DATA_DIR, "sold_cleaned.csv"), low_memory=False)
listings = pd.read_csv(os.path.join(DATA_DIR, "listings_cleaned.csv"), low_memory=False)

DATE_FIELDS = ["CloseDate", "PurchaseContractDate", "ListingContractDate", "ContractStatusChangeDate"]
for field in DATE_FIELDS:
    if field in sold.columns:
        sold[field] = pd.to_datetime(sold[field], errors="coerce")
    if field in listings.columns:
        listings[field] = pd.to_datetime(listings[field], errors="coerce")

print(f"SOLD loaded:     {len(sold):,} rows × {sold.shape[1]} columns")
print(f"LISTINGS loaded: {len(listings):,} rows × {listings.shape[1]} columns")

sold["price_ratio"] = sold["ClosePrice"] / sold["OriginalListPrice"].replace(0, np.nan)
sold["close_to_original_list_ratio"] = sold["ClosePrice"] / sold["OriginalListPrice"].replace(0, np.nan)


sold["price_per_sqft"] = sold["ClosePrice"] / sold["LivingArea"]

sold.loc[sold["LivingArea"] <= 0, "price_per_sqft"] = np.nan

sold["close_year"] = sold["CloseDate"].dt.year
sold["close_month"] = sold["CloseDate"].dt.month
sold["close_yrmo"] = sold["CloseDate"].dt.to_period("M").astype(str)

sold["listing_to_contract_days"] = (
    sold["PurchaseContractDate"] - sold["ListingContractDate"]
).dt.days

sold["contract_to_close_days"] = (
    sold["CloseDate"] - sold["PurchaseContractDate"]
).dt.days

print("\n--- Engineered Metrics (SOLD) ---")
new_cols = [
    "price_ratio", "close_to_original_list_ratio", "price_per_sqft",
    "close_year", "close_month", "close_yrmo",
    "listing_to_contract_days", "contract_to_close_days"
]
print(sold[new_cols].describe().T[["count", "mean", "min", "50%", "max"]].to_string())

listings["list_year"] = listings["ListingContractDate"].dt.year
listings["list_month"] = listings["ListingContractDate"].dt.month
listings["list_yrmo"] = listings["ListingContractDate"].dt.to_period("M").astype(str)

listings["list_price_per_sqft"] = listings["ListPrice"] / listings["LivingArea"]
listings.loc[listings["LivingArea"] <= 0, "list_price_per_sqft"] = np.nan

print("\n--- Engineered Metrics (LISTINGS) ---")
list_new_cols = ["list_year", "list_month", "list_yrmo", "list_price_per_sqft"]
list_new_cols = [c for c in list_new_cols if c in listings.columns]
print(listings[list_new_cols].describe().T[["count", "mean", "min", "50%", "max"]].to_string())


print("\n--- Sample Output: New Metric Columns (SOLD, 5 rows) ---")
sample_cols = [
    "CloseDate", "ClosePrice", "OriginalListPrice", "LivingArea",
    "DaysOnMarket", "price_ratio", "price_per_sqft", "close_yrmo",
    "listing_to_contract_days", "contract_to_close_days"
]
sample_cols = [c for c in sample_cols if c in sold.columns]
print(sold[sample_cols].dropna(subset=["price_ratio", "price_per_sqft"]).head(5).to_string(index=False))


print("\n--- Segmented Summary by PropertyType (SOLD) ---")

if "PropertySubType" in sold.columns:
    group_col = "PropertySubType"
else:
    group_col = "PropertyType"

summary_by_type = sold.groupby(group_col).agg(
    transactions=("ClosePrice", "count"),
    median_close_price=("ClosePrice", "median"),
    avg_close_price=("ClosePrice", "mean"),
    median_ppsf=("price_per_sqft", "median"),
    avg_days_on_market=("DaysOnMarket", "mean"),
    median_price_ratio=("price_ratio", "median"),
    avg_listing_to_contract=("listing_to_contract_days", "mean"),
    avg_contract_to_close=("contract_to_close_days", "mean"),
).sort_values("transactions", ascending=False)

for col in ["median_close_price", "avg_close_price", "median_ppsf"]:
    summary_by_type[col] = summary_by_type[col].apply(lambda x: f"${x:,.0f}" if pd.notna(x) else "")

print(summary_by_type.to_string())


print("\n--- Segmented Summary by CountyOrParish (SOLD) ---")

if "CountyOrParish" in sold.columns:
    summary_by_county = sold.groupby("CountyOrParish").agg(
        transactions=("ClosePrice", "count"),
        median_close_price=("ClosePrice", "median"),
        avg_close_price=("ClosePrice", "mean"),
        median_ppsf=("price_per_sqft", "median"),
        avg_days_on_market=("DaysOnMarket", "mean"),
        median_price_ratio=("price_ratio", "median"),
    ).sort_values("median_close_price", ascending=False)

    for col in ["median_close_price", "avg_close_price", "median_ppsf"]:
        summary_by_county[col] = summary_by_county[col].apply(lambda x: f"${x:,.0f}" if pd.notna(x) else "")

    print(summary_by_county.to_string())

print("\n--- Top 20 Listing Offices by Transaction Volume (SOLD) ---")

if "ListOfficeName" in sold.columns:
    office_summary = sold.groupby("ListOfficeName").agg(
        transactions=("ClosePrice", "count"),
        total_volume=("ClosePrice", "sum"),
        median_close_price=("ClosePrice", "median"),
    ).sort_values("total_volume", ascending=False).head(20)

    office_summary["total_volume"] = office_summary["total_volume"].apply(lambda x: f"${x:,.0f}")
    office_summary["median_close_price"] = office_summary["median_close_price"].apply(lambda x: f"${x:,.0f}")

    print(office_summary.to_string())

sold.to_csv(os.path.join(DATA_DIR, "sold_features.csv"), index=False)
listings.to_csv(os.path.join(DATA_DIR, "listings_features.csv"), index=False)

summary_by_type_raw = sold.groupby(group_col).agg(
    transactions=("ClosePrice", "count"),
    median_close_price=("ClosePrice", "median"),
    avg_close_price=("ClosePrice", "mean"),
    median_ppsf=("price_per_sqft", "median"),
    avg_days_on_market=("DaysOnMarket", "mean"),
    median_price_ratio=("price_ratio", "median"),
).sort_values("transactions", ascending=False)
summary_by_type_raw.to_csv(os.path.join(DATA_DIR, "summary_by_property_type.csv"))

if "CountyOrParish" in sold.columns:
    summary_by_county_raw = sold.groupby("CountyOrParish").agg(
        transactions=("ClosePrice", "count"),
        median_close_price=("ClosePrice", "median"),
        avg_close_price=("ClosePrice", "mean"),
        median_ppsf=("price_per_sqft", "median"),
        avg_days_on_market=("DaysOnMarket", "mean"),
        median_price_ratio=("price_ratio", "median"),
    ).sort_values("median_close_price", ascending=False)
    summary_by_county_raw.to_csv(os.path.join(DATA_DIR, "summary_by_county.csv"))

print(f"\n Saved: data/output/sold_features.csv ({len(sold):,} rows)")
print(f" Saved: data/output/listings_features.csv ({len(listings):,} rows)")
print(f" Saved: data/output/summary_by_property_type.csv")
print(f" Saved: data/output/summary_by_county.csv")
