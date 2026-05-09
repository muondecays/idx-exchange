# =============================================================================
# WEEK 1 – Monthly Dataset Aggregation
# IDX Exchange MLS Analytics Program
# =============================================================================
# Combines all monthly CRMLS Sold and Listing CSV files from January 2024
# through April 2026 into two unified datasets, then filters to Residential only.
#
# File naming convention: CRMLSSold202401.csv / CRMLSListing202401.csv
# =============================================================================

import pandas as pd
import os

DATA_DIR = "data/raw"
OUTPUT_DIR = "data/output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

script_months = pd.period_range(start="2024-01", end="2026-01", freq="M")

provided_months = pd.period_range(start="2026-02", end="2026-04", freq="M")

all_months = list(script_months) + list(provided_months)

print(f"Total months to load: {len(all_months)}")
print(f"  Script-pulled: Jan 2024 – Jan 2026 ({len(script_months)} months)")
print(f"  Provided CSVs: Feb 2026 – Apr 2026 ({len(provided_months)} months)")

sold_frames = []
sold_missing = []

for period in all_months:
    ym = period.strftime("%Y%m")
    filename = f"CRMLSSold{ym}.csv"
    filepath = os.path.join(DATA_DIR, filename)

    if os.path.exists(filepath):
        df = pd.read_csv(filepath, low_memory=False)
        df["source_file"] = filename
        sold_frames.append(df)
    else:
        sold_missing.append(filename)

if sold_missing:
    print(f"\n⚠️  Missing SOLD files ({len(sold_missing)}):")
    for f in sold_missing:
        print(f"   {f}")

sold_raw = pd.concat(sold_frames, ignore_index=True)
print(f"\nSOLD — rows before Residential filter: {len(sold_raw):,}")

listing_frames = []
listing_missing = []

for period in all_months:
    ym = period.strftime("%Y%m")
    filename = f"CRMLSListing{ym}.csv"
    filepath = os.path.join(DATA_DIR, filename)

    if os.path.exists(filepath):
        df = pd.read_csv(filepath, low_memory=False)
        df["source_file"] = filename
        listing_frames.append(df)
    else:
        listing_missing.append(filename)

if listing_missing:
    print(f"\n⚠️  Missing LISTING files ({len(listing_missing)}):")
    for f in listing_missing:
        print(f"   {f}")

listings_raw = pd.concat(listing_frames, ignore_index=True)
print(f"LISTINGS — rows before Residential filter: {len(listings_raw):,}")

print("\nSOLD — PropertyType breakdown:")
print(sold_raw["PropertyType"].value_counts(dropna=False))

print("\nLISTINGS — PropertyType breakdown:")
print(listings_raw["PropertyType"].value_counts(dropna=False))

sold = sold_raw[sold_raw["PropertyType"] == "Residential"].copy()
listings = listings_raw[listings_raw["PropertyType"] == "Residential"].copy()

print(f"\nSOLD — rows after Residential filter:   {len(sold):,}")
print(f"LISTINGS — rows after Residential filter: {len(listings):,}")

sold_out = os.path.join(OUTPUT_DIR, "sold_combined.csv")
listings_out = os.path.join(OUTPUT_DIR, "listings_combined.csv")

sold.to_csv(sold_out, index=False)
listings.to_csv(listings_out, index=False)

print(f"\nSaved: {sold_out}")
print(f"Saved: {listings_out}")