"""Stage 1 SOURCING v2: ensure the vision-screened RAW pool is on disk.
Manifest + screening verdicts are committed (ADR-0008); files re-download.
"""
import os, json, subprocess, time, urllib.request, urllib.parse
from . import config, db

RECORDS = os.path.join(config.DATA_DIR, "records")

def _manifest():
    with open(os.path.join(RECORDS, "raw_manifest.json")) as f:
        return json.load(f)

def verify_screened(key):
    with open(os.path.join(RECORDS, "screening_report.json")) as f:
        report = json.load(f)
    verdict = report.get(key, {})
    if verdict.get("watermark") is not False:
        raise RuntimeError(f"REFUSED: {key} not vision-verified clean "
                           f"(watermark={verdict.get('watermark')})")
    return verdict

def ensure_footage():
    manifest = _manifest()
    paths = []
    for item in config.USED_POOL:
        key = item["key"]
        verify_screened(key)
        entry = manifest.get(key)
        if not entry:
            raise RuntimeError(f"{key} missing from manifest")
        out = entry["path"]
        if not (os.path.exists(out) and os.path.getsize(out) > 100_000):
            title = entry.get("title") or item.get("title", "")
            if not title.startswith("File:"):
                raise RuntimeError(f"no Commons File title recorded for {key}")
            url = _common_url(title)
            print(f"  re-downloading {key}...")
            for attempt in range(3):
                rc = subprocess.run(["curl", "-sL", "-A", config.COMMONS_UA,
                                     "--limit-rate", config.COMMONS_LIMIT_RATE,
                                     "-o", out, url]).returncode
                if rc == 0 and os.path.getsize(out) > 100_000:
                    break
                time.sleep(config.COMMONS_RETRY_SLEEP)
        paths.append(out)
        print(f"  [ok] {key} ({os.path.getsize(out)//1048576}MB)")
    return paths

def _common_url(title):
    params = {"action": "query", "titles": title, "prop": "imageinfo",
              "iiprop": "url", "format": "json"}
    url = "https://commons.wikimedia.org/w/api.php?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": config.COMMONS_UA})
    data = json.load(urllib.request.urlopen(req, timeout=30))
    page = next(iter(data["query"]["pages"].values()))
    return page["imageinfo"][0]["url"].split("?")[0]

def ffprobe_duration(path):
    r = subprocess.run(["ffprobe", "-v", "quiet", "-print_format", "json",
                        "-show_format", path], capture_output=True, text=True)
    return float(json.loads(r.stdout)["format"]["duration"])

def run(case_id=None):
    paths = ensure_footage()
    vid = "COMP-v2-" + "-".join(p["key"] for p in config.USED_POOL)
    case = db.upsert_case(
        source="wikimedia_commons_raw", video_id=vid,
        title="When Authority Meets Instant Regret - 3 Raw Bodycam Case Files",
        agency="JSO / Jupiter PD / LeFlore County",
        location="Florida & Oklahoma", date="2017-2025",
        summary="Three-act raw-footage compilation: a traffic stop that refuses "
                "to stay routine; the most famous field-sobriety test ever "
                "recorded; a police chief who fails his own sobriety test.",
        regret="Every subject here had an off-ramp and drove right past it.",
        status="discovered")
    db.set_case_footage(case["id"], paths)
    print(f"[STAGE 1] case #{case['id']}: {case['title']}")
    return case["id"]

if __name__ == "__main__":
    run()
