import re
import requests

urls = [
    "https://vanban.chinhphu.vn/?docid=204678&pageid=27160",
    "https://vanban.chinhphu.vn/?docid=206454&pageid=27160",
]
headers = {"User-Agent": "Mozilla/5.0"}
for url in urls:
    r = requests.get(url, timeout=30, headers=headers)
    links = re.findall(r'href=["\']([^"\']+)["\']', r.text)
    for l in links:
        if "datafiles" in l and (".pdf" in l.lower() or ".doc" in l.lower()):
            print(url, "->", l)
