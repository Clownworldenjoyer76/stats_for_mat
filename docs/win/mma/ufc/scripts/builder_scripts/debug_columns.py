import requests
from bs4 import BeautifulSoup

r = requests.get('http://ufcstats.com/fighter-details/f4c49976c75c5ab2', headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
soup = BeautifulSoup(r.text, 'html.parser')
rows = soup.select('tr.b-fight-details__table-row')
row = rows[1]
cols = row.select('td')
for i, c in enumerate(cols):
    lines = [p.get_text(strip=True) for p in c.select('p')]
    raw = c.get_text(' ', strip=True)[:40]
    print(f'[{i}] raw={raw} | p_tags={lines}')
