import requests
from bs4 import BeautifulSoup
import json
import time

HEADERS = {"User-Agent": "Mozilla/5.0"}

# --- Step 1: Build name -> URL index from all letter pages ---
print("Building fighter URL index from ufcstats.com...")
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
            try:
                first = cols[0].get_text(strip=True)
                last = cols[1].get_text(strip=True)
                if first and last:
                    full_name = f"{first} {last}"
                    name_to_url[full_name] = a_tag["href"]
            except:
                pass
    except Exception as e:
        print(f"  Error on char={char}: {e}")
    time.sleep(1)

print(f"Index built: {len(name_to_url)} fighters found")

# --- Step 2: Scrape stats from a fighter page ---
def scrape_fighter_stats(url):
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        soup = BeautifulSoup(r.text, "html.parser")
        data = {}

        record_el = soup.select_one("span.b-content__title-record")
        if record_el:
            try:
                parts = record_el.get_text(strip=True).replace("Record:", "").strip().split("-")
                data["career_wins"] = int(parts[0])
                data["career_losses"] = int(parts[1])
            except:
                pass

        for li in soup.select("li.b-list__box-list-item"):
            i_tag = li.find("i")
            if not i_tag:
                continue
            label = i_tag.get_text(strip=True).rstrip(":")
            i_tag.decompose()
            value = li.get_text(strip=True)
            try:
                if label == "SLpM":
                    data["slpm"] = float(value)
                elif label == "Str. Acc.":
                    data["str_acc"] = float(value.replace("%", "")) / 100
                elif label == "SApM":
                    data["sapm"] = float(value)
                elif label == "Str. Def":
                    data["str_def"] = float(value.replace("%", "")) / 100
                elif label == "TD Acc.":
                    data["td_acc"] = float(value.replace("%", "")) / 100
                elif label == "TD Def.":
                    data["td_def"] = float(value.replace("%", "")) / 100
            except:
                pass

        return data
    except Exception as e:
        print(f"  Page error: {e}")
    return {}

# --- Step 3: Match fighters and scrape ---
with open("fighter_attributes.json") as f:
    attrs = json.load(f)

print(f"\nMatching and scraping {len(attrs)} fighters...")
updated = 0
not_matched = []

for i, (name, info) in enumerate(attrs.items()):
    if i % 100 == 0:
        print(f"  Progress: {i}/{len(attrs)}")

    url = name_to_url.get(name)

    if not url:
        # Try title-casing variations
        variations = [
            name.title(),
            " ".join(name.split()[::-1]),  # reverse order
        ]
        for v in variations:
            url = name_to_url.get(v)
            if url:
                break

    if not url:
        not_matched.append(name)
        continue

    stats = scrape_fighter_stats(url)
    if stats:
        attrs[name].update(stats)
        updated += 1

    time.sleep(1.2)

with open("fighter_attributes.json", "w") as f:
    json.dump(attrs, f, indent=2)

print(f"\nDone. Updated {updated} fighters.")
print(f"Not matched ({len(not_matched)}): {not_matched}")

for check in ["Israel Adesanya", "Bobby Green", "Jon Jones"]:
    d = attrs.get(check, {})
    print(f"\n{check}: wins={d.get('career_wins','?')} losses={d.get('career_losses','?')} slpm={d.get('slpm','?')} td_acc={d.get('td_acc','?')}")