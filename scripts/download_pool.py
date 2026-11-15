"""Download + vision-screen raw public-domain footage from Wikimedia Commons.
Manifest (committed in data/records/raw_manifest.json) drives re-downloads.
"""
import json, os, subprocess, time, urllib.request, urllib.parse

UA = "BodycamAutopilot/1.0 (documentary sourcing; contact: ops@regretoncamera.example)"
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "output", "footage")
os.makedirs(OUT, exist_ok=True)

def api(params):
    url = "https://commons.wikimedia.org/w/api.php?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    return json.load(urllib.request.urlopen(req, timeout=30))

def common_url(title):
    r = api({"action": "query", "titles": title, "prop": "imageinfo",
             "iiprop": "url|extmetadata", "format": "json"})
    page = next(iter(r["query"]["pages"].values()))
    ii = page["imageinfo"][0]
    lic = ii.get("extmetadata", {}).get("LicenseShortName", {}).get("value", "?")
    return ii["url"].split("?")[0], lic

def main():
    man = json.load(open(os.path.join(BASE, "data", "records", "raw_manifest.json")))
    for key, entry in man.items():
        out = os.path.join(BASE, entry["path"])
        if os.path.exists(out) and os.path.getsize(out) > 100_000:
            print(f"[skip] {key}")
            continue
        title = entry.get("title", "")
        if not title.startswith("File:"):
            print(f"[miss] {key}: no Commons title")
            continue
        url, lic = common_url(title)
        print(f"[get ] {key} ({lic})")
        rc = subprocess.run(["curl", "-sL", "-A", UA, "--limit-rate", "10M",
                             "-o", out, url]).returncode
        if rc != 0 or os.path.getsize(out) < 100_000:
            print("  FAILED"); time.sleep(20)
    print("done")

if __name__ == "__main__":
    main()
