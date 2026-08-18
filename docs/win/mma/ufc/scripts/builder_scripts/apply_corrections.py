import pandas as pd
import json

df = pd.read_parquet("ufc_master.parquet")
with open("name_corrections.json") as f:
    corrections = json.load(f)

new_rows = []
drop_indices = []

for idx, row in df.iterrows():
    f1, f2 = row["fighter_1"], row["fighter_2"]
    f1_fix = corrections.get(f1)
    f2_fix = corrections.get(f2)

    # Both names are fine or already corrected
    if f1_fix is None and f2_fix is None:
        continue

    # Single fighter name fix (list of 1)
    if f1_fix and len(f1_fix) == 1:
        df.at[idx, "fighter_1"] = f1_fix[0]
        f1 = f1_fix[0]
        f1_fix = None

    if f2_fix and len(f2_fix) == 1:
        df.at[idx, "fighter_2"] = f2_fix[0]
        f2 = f2_fix[0]
        f2_fix = None

    # Merged two-fighter name: need to split into correct row
    if f1_fix and len(f1_fix) == 2:
        # f1 was actually two fighters merged — fix fighter_1 and fighter_2
        row = df.loc[idx].copy()
        row["fighter_1"] = f1_fix[0]
        row["fighter_2"] = f1_fix[1]
        new_rows.append(row)
        drop_indices.append(idx)

    elif f2_fix and len(f2_fix) == 2:
        row = df.loc[idx].copy()
        row["fighter_1"] = f2_fix[0]
        row["fighter_2"] = f2_fix[1]
        new_rows.append(row)
        drop_indices.append(idx)

# Apply drops and additions
df = df.drop(index=drop_indices)
if new_rows:
    df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)

df = df.sort_values("match_date").reset_index(drop=True)

# Final check
with open("fighter_attributes.json") as f:
    attrs = json.load(f)
known = set(attrs.keys())

still_bad = df[~df["fighter_1"].isin(known) | ~df["fighter_2"].isin(known)]
print(f"Total fights: {len(df)}")
print(f"Rows still with unmatched names: {len(still_bad)}")
if len(still_bad):
    print(still_bad[["match_date","fighter_1","fighter_2"]].to_string())

df.to_parquet("ufc_master_clean.parquet", index=False)
print("Saved to ufc_master_clean.parquet")
