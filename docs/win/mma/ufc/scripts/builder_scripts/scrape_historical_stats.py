import requests
from bs4 import BeautifulSoup
import json
import time
from datetime import datetime
import pandas as pd

HEADERS = {"User-Agent": "Mozilla/5.0"}

# --- Step 1: Build name -> URL index ---
print("Building fighter URL index...")
name_to_url = {}
for char in "abcdefghijklmnopqrstuvwxyz":
    url = f"http://ufcstats.com/statistics/fighters?char={char}&page=all"
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        soup = BeautifulSoup(r.text, "html.parser")
        rows = soup.select("tr.b-statistics__table-row")
        for row in rows:
            cols = row.select("td")
            a_tag = row.select_one("a")
            if not cols or not a_tag:
                continue
            first = cols[0].get_text(strip=True)
            last = cols[1].get_text(strip=True)
            if first and last:
                name_to_url[f"{first} {last}"] = a_tag["href"]
    except Exception as e:
        print(f"  Error {char}: {e}")
    time.sleep(0.8)
print(f"Index: {len(name_to_url)} fighters")

MANUAL = {
    "Jj Aldrich": "JJ Aldrich", "Kb Bhullar": "KB Bhullar", "Tj Brown": "TJ Brown",
    "Jp Buys": "JP Buys", "Seungwoo Choi": "Seung-woo Choi", "Aj Cunningham": "AJ Cunningham",
    "Gloria De Paula": "Gloria de Paula", "Reinier De Ridder": "Reinier de Ridder",
    "Yadier Del Valle": "Yadier del Valle", "Tj Dillashaw": "TJ Dillashaw",
    "Aj Dobson": "AJ Dobson", "Felipe Dos Santos": "Felipe dos Santos",
    "Aj Fletcher": "AJ Fletcher", "Lone'Er Kavanagh": "Lone'er Kavanagh",
    "Tj Laramie": "TJ Laramie", "Changho Lee": "ChangHo Lee", "Molly Mccann": "Molly McCann",
    "Eric Mcconico": "Eric McConico", "Court Mcgee": "Court McGee",
    "Marcus Mcghee": "Marcus McGhee", "Conor Mcgregor": "Conor McGregor",
    "Rhys Mckee": "Rhys McKee", "Cory Mckenna": "Cory McKenna",
    "Terrance Mckinney": "Terrance McKinney", "Sara Mcmann": "Sara McMann",
    "Tommy Mcmillen": "Tommy McMillen", "Jackson Mcvey": "Jackson McVey",
    "Junyong Park": "Jun Yong Park", "Hyunsung Park": "Hyun Sung Park",
    "Marcos Rogerio De Lima": "Marcos Rogerio de Lima",
    "Douglas Silva De Andrade": "Douglas Silva de Andrade",
    "Cameron Vancamp": "Cameron VanCamp", "Cj Vergara": "CJ Vergara",
    "Tre'Ston Vines": "Tre'ston Vines", "Suyoung You": "SuYoung You",
    "Elizeu Zaleski Dos Santos": "Elizeu Zaleski dos Santos",
    "Germaine De Randamie": "Germaine de Randamie", "Ian Garry": "Ian Machado Garry",
    "Kai Kamaka Iii": "Kai Kamaka III", "Khalil Rountree Jr": "Khalil Rountree Jr.",
    "Bobby Green Jr": "Bobby Green Jr.", "Chase Hooper Jr": "Chase Hooper Jr.",
    "Da-Un Jung": "Da-un Jung", "Nuerdanbieke Shayilan": "Shayilan Nuerdanbieke",
}
lower_index = {k.lower(): v for k, v in name_to_url.items()}

def find_url(name):
    if name in name_to_url: return name_to_url[name]
    mapped = MANUAL.get(name)
    if mapped and mapped in name_to_url: return name_to_url[mapped]
    return lower_index.get(name.lower())

def parse_date(date_str):
    date_str = date_str.strip().replace(".", "")
    for fmt in ["%b %d, %Y", "%b %d %Y"]:
        try:
            return datetime.strptime(date_str, fmt)
        except:
            pass
    return None

def safe_int(val):
    try: return int(val.strip())
    except: return 0

def scrape_fighter_history(url):
    """
    Column mapping (confirmed):
      col[0]: result
      col[1]: fighters p[0]=this, p[1]=opponent
      col[2]: knockdowns p[0]=this, p[1]=opp  -- SKIP
      col[3]: sig strikes p[0]=landed, p[1]=attempted (both this fighter's)
      col[4]: total strikes -- SKIP
      col[5]: takedowns p[0]=landed, p[1]=attempted (both this fighter's)
      col[6]: event/date p[0]=event, p[1]=date
      col[7]: method
      col[8]: round
      col[9]: time
    """
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        soup = BeautifulSoup(r.text, "html.parser")
        fights = []
        rows = soup.select("tr.b-fight-details__table-row")

        for row in rows:
            cols = row.select("td")
            if len(cols) < 9:
                continue
            result = cols[0].get_text(strip=True).lower()
            if result not in ("win", "loss", "no contest", "draw"):
                continue

            def ptags(col_idx):
                return [p.get_text(strip=True) for p in cols[col_idx].select("p")]

            # col[3]: sig strikes landed/attempted for THIS fighter
            sig = ptags(3)
            sig_landed = safe_int(sig[0]) if len(sig) > 0 else 0
            sig_attempted = safe_int(sig[1]) if len(sig) > 1 else 0

            # col[5]: takedowns landed/attempted for THIS fighter
            tds = ptags(5)
            td_landed = safe_int(tds[0]) if len(tds) > 0 else 0
            td_attempted = safe_int(tds[1]) if len(tds) > 1 else 0

            # sig strikes ABSORBED = opponent's sig landed
            # Need opponent's col[3] p[0] — but we only have this fighter's page
            # We'll compute sapm from opponent data later; skip for now
            # Instead approximate: absorbed = attempted - landed (rough proxy)
            # Actually we can get it: the page shows BOTH fighters per row
            # col[3] p[0]=this fighter landed, p[1]=this fighter attempted
            # BUT wait — looking at Poirier page col[3]: ['109','198'] for UFC 318
            # That's 109 landed out of 198 attempted — makes sense for sig strikes
            # Opponent's sig landed not directly available per row without fight page
            # We'll skip sapm for now and compute from fight detail pages if needed

            # col[6]: date
            event_tags = ptags(6)
            date_str = event_tags[1] if len(event_tags) > 1 else ""
            fight_date = parse_date(date_str)

            # Fight duration
            round_tags = ptags(8)
            fight_round = safe_int(round_tags[0]) if round_tags else 1
            time_tags = ptags(9)
            fight_time_str = time_tags[0] if time_tags else "5:00"
            try:
                m, s = fight_time_str.split(":")
                last_round_seconds = int(m) * 60 + int(s)
            except:
                last_round_seconds = 300
            total_minutes = max(0.5, (fight_round - 1) * 5 + last_round_seconds / 60)

            if fight_date:
                fights.append({
                    "date": fight_date,
                    "result": result,
                    "sig_landed": sig_landed,
                    "sig_attempted": sig_attempted,
                    "td_landed": td_landed,
                    "td_attempted": td_attempted,
                    "minutes": total_minutes,
                })

        fights.sort(key=lambda x: x["date"])
        return fights
    except Exception as e:
        print(f"  Error: {e}")
        return []

def compute_stats_before(fights, before_date):
    prior = [f for f in fights if f["date"] < before_date]
    if not prior:
        return {}
    wins = sum(1 for f in prior if f["result"] == "win")
    losses = sum(1 for f in prior if f["result"] == "loss")
    total_minutes = sum(f["minutes"] for f in prior)
    total_sig_landed = sum(f["sig_landed"] for f in prior)
    total_sig_attempted = sum(f["sig_attempted"] for f in prior)
    total_td_landed = sum(f["td_landed"] for f in prior)
    total_td_attempted = sum(f["td_attempted"] for f in prior)

    slpm = total_sig_landed / total_minutes if total_minutes > 0 else 0
    str_acc = total_sig_landed / total_sig_attempted if total_sig_attempted > 0 else 0
    td_acc = total_td_landed / total_td_attempted if total_td_attempted > 0 else 0
    career_wr = wins / (wins + losses) if (wins + losses) > 0 else 0

    return {
        "h_career_wins": wins,
        "h_career_losses": losses,
        "h_career_fights": wins + losses,
        "h_career_wr": career_wr,
        "h_slpm": round(slpm, 4),
        "h_str_acc": round(str_acc, 4),
        "h_td_acc": round(td_acc, 4),
    }

# --- Step 2: Scrape ---
df = pd.read_parquet("ufc_master_clean.parquet")
all_fighters = list(set(pd.concat([df["fighter_1"], df["fighter_2"]]).unique()))

print(f"\nScraping fight history for {len(all_fighters)} fighters...")
fighter_history = {}
failed = []

for i, name in enumerate(all_fighters):
    if i % 100 == 0:
        print(f"  Progress: {i}/{len(all_fighters)}")
    url = find_url(name)
    if not url:
        failed.append(name)
        continue
    history = scrape_fighter_history(url)
    if history:
        fighter_history[name] = history
    else:
        failed.append(name)
    time.sleep(1.2)

serializable = {
    name: [{**f, "date": f["date"].strftime("%Y-%m-%d")} for f in fights]
    for name, fights in fighter_history.items()
}
with open("fighter_history.json", "w") as f:
    json.dump(serializable, f, indent=2)
print(f"\nScraped {len(fighter_history)} fighters. Failed: {len(failed)}")

# --- Step 3: Build lookup ---
print("Building per-fight historical stats...")
records = []
for _, row in df.iterrows():
    f1, f2 = row["fighter_1"], row["fighter_2"]
    date = row["match_date"].to_pydatetime()
    s1 = compute_stats_before(fighter_history.get(f1, []), date)
    s2 = compute_stats_before(fighter_history.get(f2, []), date)
    record = {"match_date": row["match_date"], "fighter_1": f1, "fighter_2": f2}
    for k, v in s1.items():
        record[f"f1_{k}"] = v
    for k, v in s2.items():
        record[f"f2_{k}"] = v
    records.append(record)

hist_df = pd.DataFrame(records)
hist_df.to_parquet("fighter_historical_stats.parquet", index=False)
print(f"Saved fighter_historical_stats.parquet — shape: {hist_df.shape}")

# Spot check Poirier — should have ~4.5 slpm, ~65% str acc
poirier = hist_df[hist_df["fighter_2"] == "Conor Mcgregor"].sort_values("match_date")
if len(poirier):
    row = poirier.iloc[0]
    print(f"\nSpot check Poirier before McGregor fight:")
    print(f"  career: {row.get('f1_h_career_wins','?')}-{row.get('f1_h_career_losses','?')}")
    print(f"  slpm: {row.get('f1_h_slpm','?')}")
    print(f"  str_acc: {row.get('f1_h_str_acc','?')}")
    print(f"  td_acc: {row.get('f1_h_td_acc','?')}")

print(f"\nFailed ({len(failed)}): {failed[:20]}")