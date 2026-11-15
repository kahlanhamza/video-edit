"""Global configuration — v2 (freeze-and-explain, raw footage, natural VO)."""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
OUTPUT_DIR = os.path.join(ROOT, "output")
CACHE_DIR = os.path.join(OUTPUT_DIR, "cache")
FOOTAGE_DIR = os.path.join(OUTPUT_DIR, "footage")
AUDIO_DIR = os.path.join(OUTPUT_DIR, "audio")
VIDEO_DIR = os.path.join(OUTPUT_DIR, "videos")
THUMB_DIR = os.path.join(OUTPUT_DIR, "thumbs")
QUEUE_DIR = os.path.join(OUTPUT_DIR, "queue")
PACKAGE_DIR = os.path.join(OUTPUT_DIR, "package")
DB_PATH = os.path.join(DATA_DIR, "autopilot.db")
COOKIES_PATH = os.path.join(DATA_DIR, "cookies.txt")

for d in (DATA_DIR, OUTPUT_DIR, CACHE_DIR, FOOTAGE_DIR, AUDIO_DIR,
          VIDEO_DIR, THUMB_DIR, QUEUE_DIR, PACKAGE_DIR):
    os.makedirs(d, exist_ok=True)

# ---------------------------------------------------------------- brand
CHANNEL_NAME = "REGRET ON CAMERA"
CHANNEL_HANDLE = "@RegretOnCamera"
NICHE = "police_bodycam_instant_regret"
LANGUAGE = "en"
YOUTUBE_CATEGORY_ID = "25"  # News & Politics
DEFAULT_TAGS = [
    "police bodycam", "body cam footage", "bodycam footage",
    "police body camera", "caught on camera", "police chase",
    "bodycam commentary", "instant regret", "police arrest footage",
    "body cam", "law enforcement footage",
]

# ---------------------------------------------------------------- video spec
TARGET_MIN_SEC = 900
TARGET_MAX_SEC = 1500
DEMO_MIN_SEC = 480
RES_W, RES_H, FPS = 1920, 1080, 30
CRF = 21
X264_PRESET = "veryfast"
AUDIO_RATE = 48000
FORWARD_BED_DB = -6.0          # raw audio during forward-play gaps

# ---------------------------------------------------------------- editing (ADR-0004)
REPLAY_SPEED = 0.45            # muted slow-mo under narration
FLASHBACK_SPEED = 0.40
HOOK_SEC = 3.5                 # raw catchable opener, audio ON
NARR_OFFSET = 0.3              # narration starts N sec into its replay
FREEZE_GRADE = "eq=saturation=0.88:contrast=1.05"
FLASHBACK_GRADE = ("eq=saturation=0.45:contrast=1.02,"
                   "colorbalance=bs=0.12:ms=0.05")
PUNCH = 1.07                   # static punch-in for replay/flashback
NO_CARDS = True                # ADR-0004: zero full-screen text cards

# ---------------------------------------------------------------- voice (M3)
TTS_VOICE = "kazi"
TTS_SPEED = 1.05               # continuous rate measured 140 wpm
WPM_MIN, WPM_MAX = 95, 155     # EFFECTIVE rate incl. dramatic pauses (measured
                               # 102.4 on real beats; continuous rate 140 at 1.05)

# ---------------------------------------------------------------- compliance
MANDATORY_DISCLAIMER = (
    "The footage shown in this video is public record, released to the public "
    "by the law-enforcement agencies depicted or obtained through public-records "
    "requests, and is used for documentary, educational and newsworthy purposes. "
    "All individuals shown are presumed innocent until proven guilty in a court "
    "of law. Charges, where mentioned, are allegations only and may have been "
    "dismissed, reduced or resolved. This channel does not condone violence; "
    "content is presented with commentary, analysis and context. Viewer "
    "discretion is advised: some footage may be disturbing."
)
END_OVERLAY_LINE1 = "PRESUMED INNOCENT. UNTIL PROVEN GUILTY."
END_OVERLAY_LINE2 = "REGRET ON CAMERA"

FORBIDDEN_PHRASES = [
    "criminal", "guilty of", "thug", "scumbag", "deserves", "evil",
    "monster", "loser", "idiot", "subhuman", "should be shot",
    "should be killed", "got what he",
]
SAFE_REPLACEMENTS = {
    "criminal": "the suspect", "thug": "the individual",
    "scumbag": "the individual", "idiot": "the individual",
    "monster": "the individual", "loser": "the individual",
}
# timestamp-speak patterns (owner: no robotic "at 180 seconds")
TIMESTAMP_PATTERNS = [
    r"\bat\s+\d{1,4}\s*(seconds?|secs?|minutes?|mins?)\b",
    r"\b(after|for|within|about|around)\s+(the\s+next\s+)?\d{1,4}\s*(seconds?|secs?|minutes?|mins?)\b",
    r"\bat\s+(the\s+)?\d{1,2}:\d{2}\s*(mark|stamp)?\b",
    r"\b(second|minute)\s+(one|two|three|four|five|\d{1,3})\b",
    r"\b\d{1,2}:\d{2}\b",
]

# ---------------------------------------------------------------- sourcing (M1)
# Pool = vision-screened CLEAN raw footage (data/records/raw_manifest.json +
# data/records/screening_report.json). Re-downloadable per ADR-0008.
USED_POOL = [
    {
        "key": "raw_mcneil_jax",
        "title": "Jacksonville Officer Bowers' body cam from the arrest of William McNeil Jr.",
        "role": "act1",
        "agency": "Jacksonville Sheriff's Office",
        "location": "Jacksonville, Florida",
        "date": "2025",
        "summary": "A traffic stop over phone use escalates when the driver refuses "
                   "to get out; the arrest that follows sparks national debate.",
        "regret": "Arguing the stop instead of handling it turns a ticket into cuffs.",
    },
    {
        "key": "raw_tiger_dui_2017",
        "title": "Dashboard camera video of Tiger Woods' DUI arrest, May 29, 2017",
        "role": "act2",
        "agency": "Jupiter Police Department",
        "location": "Jupiter, Florida",
        "date": "May 29, 2017",
        "summary": "A world-famous golfer is found asleep at the wheel of a damaged "
                   "Mercedes; the field-sobriety tests and the arrest that follow "
                   "played out on one dashcam.",
        "regret": "The man famous for total control of a car loses control of "
                  "the simplest test of all.",
    },
    {
        "key": "raw_okchief_dui",
        "title": "Dashcam video shows Oklahoma police chief DUI arrest in LeFlore County",
        "role": "act3",
        "agency": "LeFlore County law enforcement",
        "location": "LeFlore County, Oklahoma",
        "date": "2024",
        "summary": "A small-town police chief is pulled over and fails the field "
                   "sobriety tests he once administered himself.",
        "regret": "The man who enforced the law becomes its subject.",
    },
]
COMMONS_UA = ("BodycamAutopilot/1.0 (automated documentary sourcing; "
              "contact: ops@regretoncamera.example)")
COMMONS_LIMIT_RATE = "10M"
COMMONS_RETRY_SLEEP = 45
