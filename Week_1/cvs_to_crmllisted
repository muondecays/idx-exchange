import pandas as pd
import glob

files = sorted(glob.glob("/Users/cailing/idx_exchange/CSV_folder/CRMLSListing*.csv"))

dfs = []
for f in files:
    df = pd.read_csv(f)
    dfs.append(df)

master_listed = pd.concat(dfs, ignore_index=True)

master_listed.to_csv("CRMLSListing_Master.csv", index=False)
print(f"Rows: {len(master_listed)}")
print(f"Columns: {list(master_listed.columns)}")