# docs/win/mma/ufc/scripts/builder_scripts/fix_unmatched_fighters.py

import requests
from bs4 import BeautifulSoup
import json
import time

HEADERS = {"User-Agent": "Mozilla/5.0"}

# Build the full index first
print("Building index...")
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

# Manual mappings: our_name -> ufcstats_name
MANUAL = {
    "Jj Aldrich": "JJ Aldrich",
    "Kb Bhullar": "KB Bhullar",
    "Tj Brown": "TJ Brown",
    "Jp Buys": "JP Buys",
    "Seungwoo Choi": "Seung-woo Choi",
    "Aj Cunningham": "AJ Cunningham",
    "Gloria De Paula": "Gloria de Paula",
    "Reinier De Ridder": "Reinier de Ridder",
    "Yadier Del Valle": "Yadier del Valle",
    "Tj Dillashaw": "TJ Dillashaw",
    "Aj Dobson": "AJ Dobson",
    "Felipe Dos Santos": "Felipe dos Santos",
    "Aj Fletcher": "AJ Fletcher",
    "Lone'Er Kavanagh": "Lone'er Kavanagh",
    "Tj Laramie": "TJ Laramie",
    "Jeongyeong Lee": "Jeongyeong Lee",
    "Changho Lee": "ChangHo Lee",
    "Molly Mccann": "Molly McCann",
    "Eric Mcconico": "Eric McConico",
    "Court Mcgee": "Court McGee",
    "Marcus Mcghee": "Marcus McGhee",
    "Conor Mcgregor": "Conor McGregor",
    "Rhys Mckee": "Rhys McKee",
    "Cory Mckenna": "Cory McKenna",
    "Terrance Mckinney": "Terrance McKinney",
    "Sara Mcmann": "Sara McMann",
    "Tommy Mcmillen": "Tommy McMillen",
    "Jackson Mcvey": "Jackson McVey",
    "Marquel Mederos": "Marquel Mederos",
    "Junyong Park": "Jun Yong Park",
    "Hyunsung Park": "Hyun Sung Park",
    "Marcos Rogerio De Lima": "Marcos Rogerio de Lima",
    "Douglas Silva De Andrade": "Douglas Silva de Andrade",
    "Cameron Vancamp": "Cameron VanCamp",
    "Cj Vergara": "CJ Vergara",
    "Tre'Ston Vines": "Tre'ston Vines",
    "Suyoung You": "SuYoung You",
    "Elizeu Zaleski Dos Santos": "Elizeu Zaleski dos Santos",
    "Abdul-Rakhman Yakhyaev": "Abdul-Rakhman Yakhyaev",
    "Ian Garry": "Ian Machado Garry",
    "Germaine De Randamie": "Germaine de Randamie",
    "Khalil Rountree Jr": "Khalil Rountree Jr.",
    "Bobby Green Jr": "Bobby Green Jr.",
    "Chase Hooper Jr": "Chase Hooper Jr.",
    "Kai Kamaka Iii": "Kai Kamaka III",
    "Seung Woo Kang": "Seung Woo Kang",
    "Sung Bin Ko": "Sung Bin Ko",
    "Joo Sang Yoo": "Joo Sang Yoo",
    "Da-Un Jung": "Da-un Jung",
    "Su Mudaerji": "Su Mudaerji",
    "Nuerdanbieke Shayilan": "Shayilan Nuerdanbieke",
}

# For remaining unmatched, try fuzzy: lowercase comparison
lower_index = {k.lower(): v for k, v in name_to_url.items()}

def find_url(our_name):
    # 1. Direct
    if our_name in name_to_url:
        return name_to_url[our_name]
    # 2. Manual mapping
    mapped = MANUAL.get(our_name)
    if mapped and mapped in name_to_url:
        return name_to_url[mapped]
    # 3. Lowercase match
    url = lower_index.get(our_name.lower())
    if url:
        return url
    return None

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

with open("fighter_attributes.json") as f:
    attrs = json.load(f)

UNMATCHED = [
    'Jj Aldrich', 'Kb Bhullar', 'Tj Brown', 'Jp Buys', 'Seungwoo Choi',
    'Aj Cunningham', 'Gloria De Paula', 'Reinier De Ridder', 'Yadier Del Valle',
    'Tj Dillashaw', 'Aj Dobson', 'Felipe Dos Santos', 'Aj Fletcher',
    "Lone'Er Kavanagh", 'Tj Laramie', 'Jeongyeong Lee', 'Changho Lee',
    'Molly Mccann', 'Eric Mcconico', 'Court Mcgee', 'Marcus Mcghee',
    'Conor Mcgregor', 'Rhys Mckee', 'Cory Mckenna', 'Terrance Mckinney',
    'Sara Mcmann', 'Tommy Mcmillen', 'Jackson Mcvey', 'Marquel Mederos',
    'Junyong Park', 'Hyunsung Park', 'Marcos Rogerio De Lima',
    'Douglas Silva De Andrade', 'Cameron Vancamp', 'Cj Vergara',
    "Tre'Ston Vines", 'Suyoung You', 'Elizeu Zaleski Dos Santos',
    'Abdul-Rakhman Yakhyaev', 'Alex Munoz', 'Ali Al-Qaisi', 'Allen Frye',
    'Aori Qileng', 'Ariane Lipski', 'Beatriz Mesquita', 'Billy Goff',
    'Bobby Green', 'Bobby Green Jr', 'Brogan Walker-Sanchez', 'Charlie Radtke',
    'Chase Hooper Jr', 'Christian Duncan', 'Da-Un Jung', 'Daria Zheleznyakova',
    'Elves Brenner', 'Gabriel Green', 'Germaine De Randamie',
    'Hayisaer Maheshate', 'Heili Alateng', 'Ian Garry',
    'Ignacio Bahamondes Valle', 'Inoue Mizuki', 'J. P. Lebosnoyani',
    'Jaqualine Amorim', 'Javier Reyez', 'Joanne Calderwood',
    'Johnny Eduardo Denis', 'Joo Sang Yoo', 'Jose Medina', 'Julia Borella',
    'Kai Kamaka Iii', 'Katlyn Chookagian', 'Khalil Rountree Jr',
    'Louie Southerland', 'Luis Rodriguez', 'Mariya La Rosa', 'Marlon De Andrade',
    'Melissa Juarez', 'Mellisa Dixon', 'Mellisa Martinez', 'Michael Aswell',
    'Montserrat Conejo', 'Montserrat Rendon', 'Pat Freitas', 'Patrick Sabatini',
    'Paula Maia', 'Philip Rowe', 'Phillip Hawes', 'Rayanne Amanda', 'Rick Glenn',
    'Rong Zhu', 'Seung Woo Kang', 'Stephen Erceg', 'Su Mudaerji', 'Sung Bin Ko',
    'Timothy Cuamba', 'Vanessa Frey', 'Veronica Macedo', 'Viktoriya Dudakova',
    'Waldo Cortes-Acosta', 'Yana Kunitskaya'
]

print(f"\nProcessing {len(UNMATCHED)} unmatched fighters...")
updated = 0
still_missing = []

for name in UNMATCHED:
    url = find_url(name)
    if not url:
        still_missing.append(name)
        continue
    stats = scrape_fighter_stats(url)
    if stats:
        if name in attrs:
            attrs[name].update(stats)
        updated += 1
        print(f"  OK: {name} -> wins={stats.get('career_wins','?')} slpm={stats.get('slpm','?')}")
    else:
        still_missing.append(name)
    time.sleep(1.2)

with open("fighter_attributes.json", "w") as f:
    json.dump(attrs, f, indent=2)

print(f"\nUpdated: {updated}")
print(f"Still missing ({len(still_missing)}): {still_missing}")

# Spot checks
for check in ["Conor Mcgregor", "Jj Aldrich", "Ian Garry", "Bobby Green"]:
    d = attrs.get(check, {})
    print(f"\n{check}: wins={d.get('career_wins','?')} slpm={d.get('slpm','?')}")
