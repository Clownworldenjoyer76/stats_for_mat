# UFC LOCAL RETRAINING INSTRUCTIONS

## Step 1 — Create/refresh the working folder

Working folder:

```text
C:\UFC_TRAIN\stats_for_mat
```

PowerShell:

```powershell
New-Item -ItemType Directory -Force "C:\UFC_TRAIN" | Out-Null
Set-Location "C:\UFC_TRAIN"

if (-not (Test-Path "C:\UFC_TRAIN\stats_for_mat\.git")) {
    git clone https://github.com/Clownworldenjoyer76/stats_for_mat.git
}

Set-Location "C:\UFC_TRAIN\stats_for_mat"
git pull origin main
```

## Step 2 — Install the Python requirements

```powershell
Set-Location "C:\UFC_TRAIN\stats_for_mat"

python -m pip install -r "C:\UFC_TRAIN\stats_for_mat\requirements.txt"
python -m pip install beautifulsoup4
```

## Step 3 — Required files before starting

The repository must contain these scripts:

```text
C:\UFC_TRAIN\stats_for_mat\docs\win\mma\ufc\scripts\builder_scripts\parse_ufc_files.py
C:\UFC_TRAIN\stats_for_mat\docs\win\mma\ufc\scripts\builder_scripts\apply_corrections.py
C:\UFC_TRAIN\stats_for_mat\docs\win\mma\ufc\scripts\builder_scripts\build_features.py
C:\UFC_TRAIN\stats_for_mat\docs\win\mma\ufc\scripts\builder_scripts\train_model_weighted.py
C:\UFC_TRAIN\stats_for_mat\docs\win\mma\ufc\scripts\builder_scripts\evaluate_roi.py
```

The completed UFC event CSV files must be in:

```text
C:\UFC_TRAIN\stats_for_mat\docs\win\mma\ufc\manual_files
```

The current model-data files must exist on GitHub at:

```text
C:\UFC_TRAIN\stats_for_mat\data\model\fighter_attributes.json
C:\UFC_TRAIN\stats_for_mat\data\model\fighter_history.json
```

The name-correction file must exist at:

```text
C:\UFC_TRAIN\stats_for_mat\docs\win\mma\ufc\mappings\name_corrections.json
```

Copy the three required starting files into the working-folder root:

```powershell
Copy-Item "C:\UFC_TRAIN\stats_for_mat\data\model\fighter_attributes.json" "C:\UFC_TRAIN\stats_for_mat\fighter_attributes.json" -Force

Copy-Item "C:\UFC_TRAIN\stats_for_mat\data\model\fighter_history.json" "C:\UFC_TRAIN\stats_for_mat\fighter_history.json" -Force

Copy-Item "C:\UFC_TRAIN\stats_for_mat\docs\win\mma\ufc\mappings\name_corrections.json" "C:\UFC_TRAIN\stats_for_mat\name_corrections.json" -Force
```

Verify:

```powershell
Get-Item `
"C:\UFC_TRAIN\stats_for_mat\fighter_attributes.json", `
"C:\UFC_TRAIN\stats_for_mat\fighter_history.json", `
"C:\UFC_TRAIN\stats_for_mat\name_corrections.json"
```

## Step 4 — Point `parse_ufc_files.py` at the repository event files

`parse_ufc_files.py` expects a folder named:

```text
C:\UFC_TRAIN\stats_for_mat\UFC_Master
```

Create a junction from that folder to:

```text
C:\UFC_TRAIN\stats_for_mat\docs\win\mma\ufc\manual_files
```

PowerShell:

```powershell
Set-Location "C:\UFC_TRAIN\stats_for_mat"

if (-not (Test-Path "C:\UFC_TRAIN\stats_for_mat\UFC_Master")) {
    New-Item -ItemType Junction `
        -Path "C:\UFC_TRAIN\stats_for_mat\UFC_Master" `
        -Target "C:\UFC_TRAIN\stats_for_mat\docs\win\mma\ufc\manual_files"
}
```

Verify:

```powershell
Get-Item "C:\UFC_TRAIN\stats_for_mat\UFC_Master"
```

## Step 5 — Back up existing training outputs

Create:

```text
C:\UFC_TRAIN\stats_for_mat\training_backups
```

PowerShell:

```powershell
Set-Location "C:\UFC_TRAIN\stats_for_mat"

New-Item -ItemType Directory -Force "C:\UFC_TRAIN\stats_for_mat\training_backups" | Out-Null

$stamp = Get-Date -Format "yyyy-MM-dd_HHmmss"

$files = @(
    "ufc_master.parquet",
    "ufc_master_clean.parquet",
    "fighter_historical_stats.parquet",
    "ufc_features.parquet",
    "ufc_model.pkl",
    "test_predictions.csv",
    "fighter_history.json"
)

foreach ($file in $files) {
    $source = "C:\UFC_TRAIN\stats_for_mat\$file"
    if (Test-Path $source) {
        Copy-Item $source "C:\UFC_TRAIN\stats_for_mat\training_backups\$stamp`_$file"
    }
}
```

## Step 6 — Build `ufc_master.parquet`

Run from:

```text
C:\UFC_TRAIN\stats_for_mat
```

PowerShell:

```powershell
Set-Location "C:\UFC_TRAIN\stats_for_mat"

python "C:\UFC_TRAIN\stats_for_mat\docs\win\mma\ufc\scripts\builder_scripts\parse_ufc_files.py"
```

Output:

```text
C:\UFC_TRAIN\stats_for_mat\ufc_master.parquet
```

Verify:

```powershell
Get-Item "C:\UFC_TRAIN\stats_for_mat\ufc_master.parquet"
```

## Step 7 — Build `ufc_master_clean.parquet`

Required root files:

```text
C:\UFC_TRAIN\stats_for_mat\ufc_master.parquet
C:\UFC_TRAIN\stats_for_mat\name_corrections.json
C:\UFC_TRAIN\stats_for_mat\fighter_attributes.json
```

PowerShell:

```powershell
Set-Location "C:\UFC_TRAIN\stats_for_mat"

python "C:\UFC_TRAIN\stats_for_mat\docs\win\mma\ufc\scripts\builder_scripts\apply_corrections.py"
```

Output:

```text
C:\UFC_TRAIN\stats_for_mat\ufc_master_clean.parquet
```

Verify:

```powershell
@'
import pandas as pd

path = r"C:\UFC_TRAIN\stats_for_mat\ufc_master_clean.parquet"

df = pd.read_parquet(path)

print("ROWS:", len(df))
print("DATE MIN:", pd.to_datetime(df["match_date"]).min())
print("DATE MAX:", pd.to_datetime(df["match_date"]).max())
'@ | python -
```

## Step 8 — Update `fighter_history.json`

Required files:

```text
C:\UFC_TRAIN\stats_for_mat\fighter_history.json
C:\UFC_TRAIN\stats_for_mat\ufc_master_clean.parquet
```

Locate Edge:

```powershell
$edge = @(
    "C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    "C:\Program Files\Microsoft\Edge\Application\msedge.exe"
) | Where-Object { Test-Path $_ } | Select-Object -First 1
```

Start Edge with the UFCStats browser profile and debugging port:

```powershell
Start-Process -FilePath $edge -ArgumentList `
'--remote-debugging-port=9222', `
'--user-data-dir="C:\UFC_TRAIN\ufcstats_browser"', `
"http://ufcstats.com/statistics/fighters?char=a&page=all"
```

Verify the browser connection:

```powershell
Invoke-RestMethod "http://127.0.0.1:9222/json/version"
```

Run the `fighter_history.json` Edge update process against:

```text
C:\UFC_TRAIN\stats_for_mat\fighter_history.json
```

using:

```text
C:\UFC_TRAIN\stats_for_mat\ufc_master_clean.parquet
```

The resulting updated file must be:

```text
C:\UFC_TRAIN\stats_for_mat\fighter_history.json
```

Verify its maximum stored fight date:

```powershell
@'
import json
import pandas as pd

path = r"C:\UFC_TRAIN\stats_for_mat\fighter_history.json"

with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)

dates = []

for fights in data.values():
    for fight in fights:
        try:
            dates.append(pd.Timestamp(fight["date"]))
        except:
            pass

print("FIGHTERS:", len(data))
print("DATE MIN:", min(dates))
print("DATE MAX:", max(dates))
'@ | python -
```

## Step 9 — Rebuild `fighter_historical_stats.parquet` from `fighter_history.json`

Required files:

```text
C:\UFC_TRAIN\stats_for_mat\ufc_master_clean.parquet
C:\UFC_TRAIN\stats_for_mat\fighter_history.json
```

PowerShell:

```powershell
Set-Location "C:\UFC_TRAIN\stats_for_mat"

@'
import json
import pandas as pd

MASTER = r"C:\UFC_TRAIN\stats_for_mat\ufc_master_clean.parquet"
HISTORY = r"C:\UFC_TRAIN\stats_for_mat\fighter_history.json"
OUTPUT = r"C:\UFC_TRAIN\stats_for_mat\fighter_historical_stats.parquet"

master = pd.read_parquet(MASTER)
master["match_date"] = pd.to_datetime(master["match_date"])

with open(HISTORY, "r", encoding="utf-8") as f:
    raw = json.load(f)

history = {}

for fighter, fights in raw.items():
    parsed = []

    for fight in fights:
        try:
            parsed.append({
                **fight,
                "date": pd.Timestamp(fight["date"])
            })
        except:
            pass

    history[fighter] = sorted(parsed, key=lambda x: x["date"])

def compute_stats_before(fighter, before_date):
    prior = [
        f for f in history.get(fighter, [])
        if f["date"] < before_date
    ]

    if not prior:
        return {}

    wins = sum(1 for f in prior if f.get("result") == "win")
    losses = sum(1 for f in prior if f.get("result") == "loss")

    minutes = sum(float(f.get("minutes", 0) or 0) for f in prior)
    sig_landed = sum(int(f.get("sig_landed", 0) or 0) for f in prior)
    sig_attempted = sum(int(f.get("sig_attempted", 0) or 0) for f in prior)
    td_landed = sum(int(f.get("td_landed", 0) or 0) for f in prior)
    td_attempted = sum(int(f.get("td_attempted", 0) or 0) for f in prior)

    return {
        "h_career_wins": wins,
        "h_career_losses": losses,
        "h_career_fights": wins + losses,
        "h_career_wr": wins / (wins + losses) if wins + losses else 0,
        "h_slpm": round(sig_landed / minutes, 4) if minutes else 0,
        "h_str_acc": round(sig_landed / sig_attempted, 4) if sig_attempted else 0,
        "h_td_acc": round(td_landed / td_attempted, 4) if td_attempted else 0,
    }

records = []

for _, row in master.iterrows():
    f1 = str(row["fighter_1"])
    f2 = str(row["fighter_2"])
    date = pd.Timestamp(row["match_date"])

    s1 = compute_stats_before(f1, date)
    s2 = compute_stats_before(f2, date)

    record = {
        "match_date": date,
        "fighter_1": f1,
        "fighter_2": f2,
    }

    for key, value in s1.items():
        record[f"f1_{key}"] = value

    for key, value in s2.items():
        record[f"f2_{key}"] = value

    records.append(record)

hist = pd.DataFrame(records)

hist.to_parquet(OUTPUT, index=False)

check = pd.read_parquet(OUTPUT)

print("ROWS:", len(check))
print("DATE MIN:", check["match_date"].min())
print("DATE MAX:", check["match_date"].max())

if len(check) != len(master):
    raise RuntimeError(
        f"ROW COUNT MISMATCH: master={len(master)} historical={len(check)}"
    )
'@ | python -
```

Output:

```text
C:\UFC_TRAIN\stats_for_mat\fighter_historical_stats.parquet
```

## Step 10 — Build `ufc_features.parquet`

Required files:

```text
C:\UFC_TRAIN\stats_for_mat\ufc_master_clean.parquet
C:\UFC_TRAIN\stats_for_mat\fighter_attributes.json
C:\UFC_TRAIN\stats_for_mat\fighter_historical_stats.parquet
```

PowerShell:

```powershell
Set-Location "C:\UFC_TRAIN\stats_for_mat"

python "C:\UFC_TRAIN\stats_for_mat\docs\win\mma\ufc\scripts\builder_scripts\build_features.py"
```

Output:

```text
C:\UFC_TRAIN\stats_for_mat\ufc_features.parquet
```

Verify:

```powershell
@'
import pandas as pd

path = r"C:\UFC_TRAIN\stats_for_mat\ufc_features.parquet"

df = pd.read_parquet(path)

print("ROWS:", len(df))
print("COLUMNS:", len(df.columns))
print("DATE MIN:", pd.to_datetime(df["match_date"]).min())
print("DATE MAX:", pd.to_datetime(df["match_date"]).max())
'@ | python -
```

## Step 11 — Train the model

Required file:

```text
C:\UFC_TRAIN\stats_for_mat\ufc_features.parquet
```

PowerShell:

```powershell
Set-Location "C:\UFC_TRAIN\stats_for_mat"

python "C:\UFC_TRAIN\stats_for_mat\docs\win\mma\ufc\scripts\builder_scripts\train_model_weighted.py"
```

Outputs:

```text
C:\UFC_TRAIN\stats_for_mat\ufc_model.pkl
C:\UFC_TRAIN\stats_for_mat\test_predictions.csv
```

Verify:

```powershell
Get-Item `
"C:\UFC_TRAIN\stats_for_mat\ufc_model.pkl", `
"C:\UFC_TRAIN\stats_for_mat\test_predictions.csv"
```

## Step 12 — Run the evaluation

Required files:

```text
C:\UFC_TRAIN\stats_for_mat\ufc_model.pkl
C:\UFC_TRAIN\stats_for_mat\ufc_features.parquet
```

PowerShell:

```powershell
Set-Location "C:\UFC_TRAIN\stats_for_mat"

python "C:\UFC_TRAIN\stats_for_mat\docs\win\mma\ufc\scripts\builder_scripts\evaluate_roi.py"
```

Output:

```text
C:\UFC_TRAIN\stats_for_mat\test_predictions.csv
```

## Step 13 — Copy the five files required by the GitHub UFC pipeline

Create:

```text
C:\UFC_TRAIN\stats_for_mat\data\model
```

PowerShell:

```powershell
New-Item -ItemType Directory -Force "C:\UFC_TRAIN\stats_for_mat\data\model" | Out-Null

Copy-Item "C:\UFC_TRAIN\stats_for_mat\fighter_attributes.json" `
"C:\UFC_TRAIN\stats_for_mat\data\model\fighter_attributes.json" -Force

Copy-Item "C:\UFC_TRAIN\stats_for_mat\fighter_history.json" `
"C:\UFC_TRAIN\stats_for_mat\data\model\fighter_history.json" -Force

Copy-Item "C:\UFC_TRAIN\stats_for_mat\fighter_historical_stats.parquet" `
"C:\UFC_TRAIN\stats_for_mat\data\model\fighter_historical_stats.parquet" -Force

Copy-Item "C:\UFC_TRAIN\stats_for_mat\ufc_master_clean.parquet" `
"C:\UFC_TRAIN\stats_for_mat\data\model\ufc_master_clean.parquet" -Force

Copy-Item "C:\UFC_TRAIN\stats_for_mat\ufc_model.pkl" `
"C:\UFC_TRAIN\stats_for_mat\data\model\ufc_model.pkl" -Force
```

Verify all five:

```powershell
Get-Item `
"C:\UFC_TRAIN\stats_for_mat\data\model\fighter_attributes.json", `
"C:\UFC_TRAIN\stats_for_mat\data\model\fighter_history.json", `
"C:\UFC_TRAIN\stats_for_mat\data\model\fighter_historical_stats.parquet", `
"C:\UFC_TRAIN\stats_for_mat\data\model\ufc_master_clean.parquet", `
"C:\UFC_TRAIN\stats_for_mat\data\model\ufc_model.pkl" |
Select-Object FullName,Length,LastWriteTime
```

## Final local training files

```text
C:\UFC_TRAIN\stats_for_mat\fighter_attributes.json
C:\UFC_TRAIN\stats_for_mat\fighter_history.json
C:\UFC_TRAIN\stats_for_mat\fighter_historical_stats.parquet
C:\UFC_TRAIN\stats_for_mat\name_corrections.json
C:\UFC_TRAIN\stats_for_mat\ufc_master.parquet
C:\UFC_TRAIN\stats_for_mat\ufc_master_clean.parquet
C:\UFC_TRAIN\stats_for_mat\ufc_features.parquet
C:\UFC_TRAIN\stats_for_mat\ufc_model.pkl
C:\UFC_TRAIN\stats_for_mat\test_predictions.csv
```

## Final GitHub pipeline model files

```text
C:\UFC_TRAIN\stats_for_mat\data\model\fighter_attributes.json
C:\UFC_TRAIN\stats_for_mat\data\model\fighter_history.json
C:\UFC_TRAIN\stats_for_mat\data\model\fighter_historical_stats.parquet
C:\UFC_TRAIN\stats_for_mat\data\model\ufc_master_clean.parquet
C:\UFC_TRAIN\stats_for_mat\data\model\ufc_model.pkl
```
