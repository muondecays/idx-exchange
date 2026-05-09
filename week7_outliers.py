# =============================================================================
# WEEK 7 – Outlier Detection and Data Quality
# IDX Exchange MLS Analytics Program
# =============================================================================
# Uses IQR method to flag extreme values in key numeric fields.
# Does not delete records outright, adds flag columns and saves two versions:
#   1. Full flagged dataset (all rows, with outlier flags)
#   2. Clean filtered dataset (outliers removed, for Tableau use)
#
# Also normalizes known brokerage name variants (e.g. Compass vs COMPASS)
#
# Input:  data/output/sold_features.csv
#         data/output/listings_features.csv
# Output: data/output/sold_flagged.csv       ← full dataset with flags
#         data/output/sold_final.csv         ← outliers removed (Tableau-ready)
#         data/output/listings_flagged.csv
#         data/output/listings_final.csv
# =============================================================================

import pandas as pd
import numpy as np
import os

DATA_DIR = "data/output"


sold = pd.read_csv(os.path.join(DATA_DIR, "sold_features.csv"), low_memory=False)
listings = pd.read_csv(os.path.join(DATA_DIR, "listings_features.csv"), low_memory=False)


DATE_FIELDS = ["CloseDate", "PurchaseContractDate", "ListingContractDate", "ContractStatusChangeDate"]
for field in DATE_FIELDS:
    if field in sold.columns:
        sold[field] = pd.to_datetime(sold[field], errors="coerce")
    if field in listings.columns:
        listings[field] = pd.to_datetime(listings[field], errors="coerce")

print(f"SOLD loaded:     {len(sold):,} rows × {sold.shape[1]} columns")
print(f"LISTINGS loaded: {len(listings):,} rows × {listings.shape[1]} columns")


OFFICE_NAME_MAP = {
    "COMPASS": "Compass",
    "Berkshire Hathaway HomeService": "Berkshire Hathaway HomeServices California Properties",
    "eXp Realty of California, Inc.": "eXp Realty of California Inc",
}

if "ListOfficeName" in sold.columns:
    sold["ListOfficeName"] = sold["ListOfficeName"].replace(OFFICE_NAME_MAP)
if "BuyerOfficeName" in sold.columns:
    sold["BuyerOfficeName"] = sold["BuyerOfficeName"].replace(OFFICE_NAME_MAP)
if "ListOfficeName" in listings.columns:
    listings["ListOfficeName"] = listings["ListOfficeName"].replace(OFFICE_NAME_MAP)

print("\nBrokerage name variants normalized.")


def flag_iqr_outliers(df, col, multiplier=1.5):
    """
    Flags records outside [Q1 - multiplier*IQR, Q3 + multiplier*IQR].
    Adds a boolean flag column: flag_outlier_{col}
    Returns (df, lower_bound, upper_bound, n_flagged)
    """
    if col not in df.columns:
        return df, None, None, 0

    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - multiplier * iqr
    upper = q3 + multiplier * iqr

    flag_col = f"flag_outlier_{col}"
    df[flag_col] = df[col].notna() & ((df[col] < lower) | (df[col] > upper))
    n_flagged = df[flag_col].sum()

    return df, lower, upper, n_flagged


SOLD_OUTLIER_FIELDS = [
    "ClosePrice",
    "LivingArea",
    "DaysOnMarket",
    "price_per_sqft",
    "price_ratio",
    "listing_to_contract_days",
    "contract_to_close_days",
    "LotSizeAcres",
]

print("\n--- SOLD — IQR Outlier Flagging ---")
print(f"{'Field':<30} {'Lower':>15} {'Upper':>15} {'Flagged':>10} {'% of total':>12}")
print("-" * 85)

for col in SOLD_OUTLIER_FIELDS:
    before = len(sold)
    sold, lower, upper, n_flagged = flag_iqr_outliers(sold, col)
    if lower is not None:
        print(f"{col:<30} {lower:>15,.2f} {upper:>15,.2f} {n_flagged:>10,} {n_flagged/before*100:>11.2f}%")


LISTINGS_OUTLIER_FIELDS = [
    "ListPrice",
    "LivingArea",
    "DaysOnMarket",
    "list_price_per_sqft",
    "LotSizeAcres",
]

print("\n--- LISTINGS — IQR Outlier Flagging ---")
print(f"{'Field':<30} {'Lower':>15} {'Upper':>15} {'Flagged':>10} {'% of total':>12}")
print("-" * 85)

for col in LISTINGS_OUTLIER_FIELDS:
    before = len(listings)
    listings, lower, upper, n_flagged = flag_iqr_outliers(listings, col)
    if lower is not None:
        print(f"{col:<30} {lower:>15,.2f} {upper:>15,.2f} {n_flagged:>10,} {n_flagged/before*100:>11.2f}%")


sold_outlier_flags = [c for c in sold.columns if c.startswith("flag_outlier_")]
listings_outlier_flags = [c for c in listings.columns if c.startswith("flag_outlier_")]

core_flags_sold = ["flag_outlier_ClosePrice", "flag_outlier_LivingArea", "flag_outlier_DaysOnMarket"]
core_flags_listings = ["flag_outlier_ListPrice", "flag_outlier_LivingArea", "flag_outlier_DaysOnMarket"]

sold["flag_any_outlier"] = sold[[c for c in core_flags_sold if c in sold.columns]].any(axis=1)
listings["flag_any_outlier"] = listings[[c for c in core_flags_listings if c in listings.columns]].any(axis=1)

print(f"\nSOLD — records flagged as outlier on ANY field: {sold['flag_any_outlier'].sum():,} "
      f"({sold['flag_any_outlier'].sum()/len(sold)*100:.1f}%)")
print(f"LISTINGS — records flagged as outlier on ANY field: {listings['flag_any_outlier'].sum():,} "
      f"({listings['flag_any_outlier'].sum()/len(listings)*100:.1f}%)")


sold["flag_business_rule_invalid"] = (
    (sold["ClosePrice"] <= 0) |
    (sold["LivingArea"] <= 0) |
    (sold["DaysOnMarket"] < 0) |
    (sold["price_ratio"] <= 0)
)

listings["flag_business_rule_invalid"] = (
    (listings["ListPrice"] <= 0) |
    (listings["LivingArea"] <= 0) |
    (listings["DaysOnMarket"] < 0)
)

print(f"\nSOLD — business rule violations: {sold['flag_business_rule_invalid'].sum():,}")
print(f"LISTINGS — business rule violations: {listings['flag_business_rule_invalid'].sum():,}")


def compare_stats(df_full, df_clean, col, label):
    if col not in df_full.columns:
        return
    before_median = df_full[col].median()
    before_mean = df_full[col].mean()
    after_median = df_clean[col].median()
    after_mean = df_clean[col].mean()
    print(f"  {col}:")
    print(f"    Before — median: {before_median:>12,.2f}  mean: {before_mean:>12,.2f}  n={len(df_full):,}")
    print(f"    After  — median: {after_median:>12,.2f}  mean: {after_mean:>12,.2f}  n={len(df_clean):,}")

sold_clean = sold[
    ~sold["flag_any_outlier"] &
    ~sold["flag_business_rule_invalid"]
].copy()

listings_clean = listings[
    ~listings["flag_any_outlier"] &
    ~listings["flag_business_rule_invalid"]
].copy()

print(f"\n--- Before/After Comparison (SOLD) ---")
for col in ["ClosePrice", "LivingArea", "DaysOnMarket", "price_per_sqft", "price_ratio"]:
    compare_stats(sold, sold_clean, col, "SOLD")

print(f"\n--- Dataset Size Comparison ---")
print(f"SOLD   — full: {len(sold):,}  →  clean: {len(sold_clean):,}  "
      f"(removed {len(sold)-len(sold_clean):,} rows, {(len(sold)-len(sold_clean))/len(sold)*100:.1f}%)")
print(f"LISTINGS — full: {len(listings):,}  →  clean: {len(listings_clean):,}  "
      f"(removed {len(listings)-len(listings_clean):,} rows, {(len(listings)-len(listings_clean))/len(listings)*100:.1f}%)")

sold.to_csv(os.path.join(DATA_DIR, "sold_flagged.csv"), index=False)
sold_clean.to_csv(os.path.join(DATA_DIR, "sold_final.csv"), index=False)

listings.to_csv(os.path.join(DATA_DIR, "listings_flagged.csv"), index=False)
listings_clean.to_csv(os.path.join(DATA_DIR, "listings_final.csv"), index=False)

print(f"\n  Saved: data/output/sold_flagged.csv     ({len(sold):,} rows — full with flags)")
print(f"  Saved: data/output/sold_final.csv       ({len(sold_clean):,} rows — Tableau-ready)")
print(f"  Saved: data/output/listings_flagged.csv ({len(listings):,} rows — full with flags)")
print(f"  Saved: data/output/listings_final.csv   ({len(listings_clean):,} rows — Tableau-ready)")
print("\nReady for Weeks 8–10 — Tableau Dashboard Development.")