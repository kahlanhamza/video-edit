"""SQLite state layer: cases + videos."""
import sqlite3, os, json, time
from . import config

def _conn():
    c = sqlite3.connect(config.DB_PATH)
    c.row_factory = sqlite3.Row
    c.execute("""CREATE TABLE IF NOT EXISTS cases(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        source TEXT, video_id TEXT UNIQUE, title TEXT, agency TEXT,
        location TEXT, date TEXT, summary TEXT, regret TEXT,
        footage_paths TEXT, status TEXT DEFAULT 'discovered',
        created_at REAL, updated_at REAL)""")
    c.execute("""CREATE TABLE IF NOT EXISTS videos(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        case_id INTEGER, title TEXT, status TEXT DEFAULT 'queued',
        video_path TEXT, thumb_path TEXT, duration REAL,
        metadata_json TEXT, youtube_id TEXT, created_at REAL)""")
    return c

def upsert_case(source, video_id, title="", agency="", location="", date="",
                summary="", regret="", status="discovered"):
    c = _conn()
    c.execute("""INSERT INTO cases(source,video_id,title,agency,location,date,
                 summary,regret,status,created_at,updated_at)
                 VALUES(?,?,?,?,?,?,?,?,?,?,?)
                 ON CONFLICT(video_id) DO UPDATE SET
                   title=excluded.title, agency=excluded.agency,
                   updated_at=excluded.updated_at""",
              (source, video_id, title, agency, location, date, summary,
               regret, status, time.time(), time.time()))
    c.commit()
    row = c.execute("SELECT * FROM cases WHERE video_id=?", (video_id,)).fetchone()
    c.close()
    return dict(row)

def get_case(case_id):
    c = _conn()
    row = c.execute("SELECT * FROM cases WHERE id=?", (case_id,)).fetchone()
    c.close()
    return dict(row) if row else None

def set_case_status(case_id, status):
    c = _conn()
    c.execute("UPDATE cases SET status=?, updated_at=? WHERE id=?",
              (status, time.time(), case_id))
    c.commit(); c.close()

def set_case_footage(case_id, paths):
    c = _conn()
    c.execute("UPDATE cases SET footage_paths=?, status='footage_ready', updated_at=? WHERE id=?",
              ("|".join(paths), time.time(), case_id))
    c.commit(); c.close()

def add_video(case_id, title, video_path, thumb_path, duration, metadata):
    c = _conn()
    c.execute("""INSERT INTO videos(case_id,title,status,video_path,thumb_path,
                 duration,metadata_json,created_at) VALUES(?,?,?,?,?,?,?,?)""",
              (case_id, title, "queued", video_path, thumb_path, duration,
               json.dumps(metadata, ensure_ascii=False), time.time()))
    c.commit()
    vid = c.execute("SELECT last_insert_rowid()").fetchone()[0]
    c.close()
    return vid

def list_videos(status=None):
    c = _conn()
    if status:
        rows = c.execute("SELECT * FROM videos WHERE status=? ORDER BY id", (status,)).fetchall()
    else:
        rows = c.execute("SELECT * FROM videos ORDER BY id").fetchall()
    c.close()
    out = []
    for r in rows:
        d = dict(r)
        if d.get("metadata_json"):
            d["metadata"] = json.loads(d["metadata_json"])
        out.append(d)
    return out

def set_video_status(video_id, status, youtube_id=None):
    c = _conn()
    c.execute("UPDATE videos SET status=?, youtube_id=? WHERE id=?",
              (status, youtube_id, video_id))
    c.commit(); c.close()
