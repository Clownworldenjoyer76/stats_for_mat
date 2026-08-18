import pandas as pd
import json

df = pd.read_parquet("ufc_master.parquet")
with open("fighter_attributes.json") as f:
    attrs = json.load(f)

known = set(attrs.keys())

# Build a lookup of all known first tokens -> full name
# e.g. "Junior" -> ["Junior Dos Santos"], "Abdul" -> ["Abdul Razak Alhassan"]
from collections import defaultdict
first_token = defaultdict(list)
for name in known:
    first_token[name.split()[0]].append(name)

def fix_row(row):
    f1, f2 = row["fighter_1"], row["fighter_2"]
    if f1 in known and f2 in known:
        return row  # both fine

    # Try to reconstruct by combining tokens from f1 and f2
    tokens = f1.split() + f2.split()

    best = None
    for i in range(1, len(tokens)):
        candidate1 = " ".join(tokens[:i])
        candidate2 = " ".join(tokens[i:])
        if candidate1 in known and candidate2 in known:
            best = (candidate1, candidate2)
            break

    if best:
        row = row.copy()
        row["fighter_1"], row["fighter_2"] = best
    return row

fixed = df.apply(fix_row, axis=1)

# Check results
still_bad_1 = ~fixed["fighter_1"].isin(known)
still_bad_2 = ~fixed["fighter_2"].isin(known)
print(f"Fixed: {(still_bad_1 | still_bad_2).sum()} rows still have unmatched names")
print("Remaining unmatched fighter_1:", fixed.loc[still_bad_1, "fighter_1"].unique().tolist())
print("Remaining unmatched fighter_2:", fixed.loc[still_bad_2, "fighter_2"].unique().tolist())

fixed.to_parquet("ufc_master_clean.parquet", index=False)
print("Saved to ufc_master_clean.parquet")
