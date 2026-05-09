# =============================================================================
# WEEKS 4–5 – Data Cleaning and Preparation
# IDX Exchange MLS Analytics Program
# =============================================================================
# This script takes the enriched datasets from Weeks 2-3 and applies
# systematic cleaning transformations to produce analysis-ready datasets.
#
# Transformations:
#   - Convert date fields to datetime
#   - Cast numeric fields to correct types
#   - Flag/remove invalid numeric values
#   - Date consistency checks (flagging out-of-order dates)
#   - Geographic data quality checks
#
# Input:  data/output/sold_enriched.csv
#         data/output/listings_enriched.csv
# Output: data/output/sold_cleaned.csv
#         data/output/listings_cleaned.csv
# =============================================================================

import pandas as pd
import numpy as np
import os

DATA_DIR = "data/output"


sold = pd.read_csv(os.path.join(DATA_DIR, "sold_enriched.csv"), low_memory=False)
listings = pd.read_csv(os.path.join(DATA_DIR, "listings_enriched.csv"), low_memory=False)

print(f"SOLD loaded:     {len(sold):,} rows × {sold.shape[1]} columns")
print(f"LISTINGS loaded: {len(listings):,} rows × {listings.shape[1]} columns")


def drop_high_missing(df, label, threshold=0.90):
    """
    Drops columns where more than `threshold` fraction of values are null.
    Keeps core analytics fields regardless of missingness.
    """

    core_fields = [
        "ClosePrice", "ListPrice", "OriginalListPrice", "LivingArea",
        "DaysOnMarket", "BedroomsTotal", "BathroomsTotalInteger",
        "CloseDate", "ListingContractDate", "PurchaseContractDate",
        "ContractStatusChangeDate", "CountyOrParish", "PostalCode",
        "MLSAreaMajor", "ListOfficeName", "BuyerOfficeName",
        "Latitude", "Longitude", "PropertySubType", "YearBuilt",
        "LotSizeAcres", "year_month", "rate_30yr_fixed"
    ]

    total = len(df)
    missing_pct = df.isnull().sum() / total
    high_missing = missing_pct[missing_pct > threshold].index.tolist()


    to_drop = [col for col in high_missing if col not in core_fields]

    print(f"\n{label} — Dropping {len(to_drop)} columns with >{threshold*100:.0f}% missing")
    if to_drop:
        print(f"  Dropped: {to_drop[:10]}{'...' if len(to_drop) > 10 else ''}")

    return df.drop(columns=to_drop)

sold = drop_high_missing(sold, "SOLD")
listings = drop_high_missing(listings, "LISTINGS")


DATE_FIELDS = [
    "CloseDate",
    "PurchaseContractDate",
    "ListingContractDate",
    "ContractStatusChangeDate"
]

print("\n--- Converting date fields ---")
for field in DATE_FIELDS:
    if field in sold.columns:
        before = sold[field].dtype
        sold[field] = pd.to_datetime(sold[field], errors="coerce")
        print(f"  SOLD {field}: {before} → {sold[field].dtype}")

    if field in listings.columns:
        before = listings[field].dtype
        listings[field] = pd.to_datetime(listings[field], errors="coerce")
        print(f"  LISTINGS {field}: {before} → {listings[field].dtype}")


NUMERIC_FIELDS = [
    "ClosePrice", "ListPrice", "OriginalListPrice",
    "LivingArea", "LotSizeAcres",
    "BedroomsTotal", "BathroomsTotalInteger",
    "DaysOnMarket", "YearBuilt",
    "Latitude", "Longitude"
]

print("\n--- Casting numeric fields ---")
for field in NUMERIC_FIELDS:
    if field in sold.columns:
        sold[field] = pd.to_numeric(sold[field], errors="coerce")
    if field in listings.columns:
        listings[field] = pd.to_numeric(listings[field], errors="coerce")

print("  Done.")


def flag_invalid_numerics(df, label):
    """
    Adds boolean flag columns for records with logically invalid numeric values.
    Returns a summary of how many records were flagged per rule.
    """
    flags = {}

    if "ClosePrice" in df.columns:
        df["flag_invalid_close_price"] = df["ClosePrice"].notna() & (df["ClosePrice"] <= 0)
        flags["ClosePrice <= 0"] = df["flag_invalid_close_price"].sum()

    if "LivingArea" in df.columns:
        df["flag_invalid_living_area"] = df["LivingArea"].notna() & (df["LivingArea"] <= 0)
        flags["LivingArea <= 0"] = df["flag_invalid_living_area"].sum()

    if "DaysOnMarket" in df.columns:
        df["flag_negative_dom"] = df["DaysOnMarket"].notna() & (df["DaysOnMarket"] < 0)
        flags["DaysOnMarket < 0"] = df["flag_negative_dom"].sum()

    if "BedroomsTotal" in df.columns:
        df["flag_negative_beds"] = df["BedroomsTotal"].notna() & (df["BedroomsTotal"] < 0)
        flags["BedroomsTotal < 0"] = df["flag_negative_beds"].sum()

    if "BathroomsTotalInteger" in df.columns:
        df["flag_negative_baths"] = df["BathroomsTotalInteger"].notna() & (df["BathroomsTotalInteger"] < 0)
        flags["BathroomsTotalInteger < 0"] = df["flag_negative_baths"].sum()

    print(f"\n{label} — Invalid Numeric Flags:")
    for rule, count in flags.items():
        print(f"  {rule}: {count:,} records flagged")

    return df

sold = flag_invalid_numerics(sold, "SOLD")
listings = flag_invalid_numerics(listings, "LISTINGS")


def flag_date_consistency(df, label):
    """
    Flags records where dates appear in an illogical order.
    Expected order: ListingContractDate → PurchaseContractDate → CloseDate
    """
    print(f"\n{label} — Date Consistency Checks:")

    if "ListingContractDate" in df.columns and "CloseDate" in df.columns:
        mask = df["ListingContractDate"].notna() & df["CloseDate"].notna()
        df["listing_after_close_flag"] = False
        df.loc[mask, "listing_after_close_flag"] = (
            df.loc[mask, "ListingContractDate"] > df.loc[mask, "CloseDate"]
        )
        print(f"  listing_after_close_flag:   {df['listing_after_close_flag'].sum():,}")

    if "PurchaseContractDate" in df.columns and "CloseDate" in df.columns:
        mask = df["PurchaseContractDate"].notna() & df["CloseDate"].notna()
        df["purchase_after_close_flag"] = False
        df.loc[mask, "purchase_after_close_flag"] = (
            df.loc[mask, "PurchaseContractDate"] > df.loc[mask, "CloseDate"]
        )
        print(f"  purchase_after_close_flag:  {df['purchase_after_close_flag'].sum():,}")

    if "ListingContractDate" in df.columns and "PurchaseContractDate" in df.columns:
        mask = df["ListingContractDate"].notna() & df["PurchaseContractDate"].notna()
        df["negative_timeline_flag"] = False
        df.loc[mask, "negative_timeline_flag"] = (
            df.loc[mask, "ListingContractDate"] > df.loc[mask, "PurchaseContractDate"]
        )
        print(f"  negative_timeline_flag:     {df['negative_timeline_flag'].sum():,}")

    return df

sold = flag_date_consistency(sold, "SOLD")
listings = flag_date_consistency(listings, "LISTINGS")


def flag_geo_issues(df, label):
    """
    Flags records with missing, zero, or implausible coordinates.
    California longitudes should be negative (roughly -114 to -124).
    """
    print(f"\n{label} — Geographic Data Quality:")

    if "Latitude" not in df.columns or "Longitude" not in df.columns:
        print("  Latitude/Longitude columns not found — skipping.")
        return df

    df["flag_missing_coords"] = df["Latitude"].isnull() | df["Longitude"].isnull()
    print(f"  Missing coordinates:         {df['flag_missing_coords'].sum():,}")

    df["flag_zero_coords"] = (df["Latitude"] == 0) | (df["Longitude"] == 0)
    print(f"  Zero coordinate (sentinel):  {df['flag_zero_coords'].sum():,}")

    df["flag_positive_longitude"] = df["Longitude"].notna() & (df["Longitude"] > 0)
    print(f"  Positive longitude (invalid for CA): {df['flag_positive_longitude'].sum():,}")

    valid_lat = df["Latitude"].between(32.5, 42.0)
    valid_lon = df["Longitude"].between(-124.5, -114.0)
    df["flag_out_of_state_coords"] = (
        df["Latitude"].notna() & df["Longitude"].notna() &
        (~valid_lat | ~valid_lon)
    )
    print(f"  Out-of-state/implausible:    {df['flag_out_of_state_coords'].sum():,}")

    return df

sold = flag_geo_issues(sold, "SOLD")
listings = flag_geo_issues(listings, "LISTINGS")


print("\nFinal Dataset Summary: ")
print(f"SOLD:     {len(sold):,} rows × {sold.shape[1]} columns")
print(f"LISTINGS: {len(listings):,} rows × {listings.shape[1]} columns")


for field in DATE_FIELDS:
    if field in sold.columns:
        print(f"  SOLD {field} dtype: {sold[field].dtype}")


flag_cols = [c for c in sold.columns if c.startswith("flag_")]
print(f"\nSOLD — all flag columns:")
for col in flag_cols:
    print(f"  {col}: {sold[col].sum():,} flagged")


sold.to_csv(os.path.join(DATA_DIR, "sold_cleaned.csv"), index=False)
listings.to_csv(os.path.join(DATA_DIR, "listings_cleaned.csv"), index=False)

print(f"\n Saved: data/output/sold_cleaned.csv ({len(sold):,} rows)")
print(f" Saved: data/output/listings_cleaned.csv ({len(listings):,} rows)")
print("\nReady for Week 6 — Feature Engineering.")