import pandas as pd
import glob

files = sorted(glob.glob("/Users/cailing/idx_exchange/CSV_folder/CRMLSSold*.csv"))

dfs = []
for f in files:
    dfs.append(pd.read_csv(f))

master_sold = pd.concat(dfs, ignore_index=True)

master_sold.to_csv("CRMLSSold_Master.csv", index=False)
print(f"Total rows: {len(master_sold)}")
print(f"Columns: {list(master_sold.columns)}")