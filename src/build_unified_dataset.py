# Add this near the top of the file
TEAM_FIXES = {
    # Correct broken or truncated
    "Pittsb": "Pittsburgh Steelers",
    # Short/abbreviated aliases
    "LA Rams": "Los Angeles Rams",
    "Oakland": "Las Vegas Raiders",
    "SD Chargers": "Los Angeles Chargers",
    "San Diego Chargers": "Los Angeles Chargers",
    "NY Jets": "New York Jets",
    "Washington": "Washington Commanders",
    "Jax Jaguars": "Jacksonville Jaguars",
    "Tampa Bay Bucs": "Tampa Bay Buccaneers",
    "NE Patriots": "New England Patriots",
}

def clean_team_name(name: str) -> str:
    """Standardize known variants and fix truncations."""
    if not isinstance(name, str):
        name = str(name)
    n = name.strip()
    if n in TEAM_FIXES:
        return TEAM_FIXES[n]
    # fallback capitalization
    return " ".join(w.capitalize() for w in n.split())

def load_and_prepare(file_path: Path) -> pd.DataFrame | None:
    df = pd.read_csv(file_path)
    team_col = next((c for c in df.columns if 'team' in c.lower()), None)
    if team_col is None:
        print(f"[SKIP] {file_path.name} (no team column)")
        return None

    df = df.rename(columns={team_col: "team"})
    df["team"] = df["team"].astype(str).apply(clean_team_name)
    df = df.drop_duplicates(subset=["team"], keep="first")

    prefix = file_path.stem.lower()
    rename_map = {c: f"{prefix}_{c}" for c in df.columns if c != "team"}
    df = df.rename(columns=rename_map)

    # Convert to numeric
    for c in df.columns:
        if c != "team":
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    return df
