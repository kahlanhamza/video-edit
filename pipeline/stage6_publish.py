"""Stage 6 PUBLISHING: 1-click human review queue. NO auto-upload."""
import os, json, sys
from . import config, db

def queue_for_review(case_id, title, video_path, thumb_path, duration, metadata):
    vid = db.add_video(case_id, title, video_path, thumb_path, duration, metadata)
    entry = {
        "video_id": vid,
        "case_id": case_id,
        "title": title,
        "video_path": video_path,
        "thumb_path": thumb_path,
        "duration_sec": round(duration, 1),
        "metadata": metadata,
        "status": "queued_for_review",
        "review_note": "Owner reviews title/description/thumbnail, then runs: "
                       "python3 -m pipeline.stage6_publish approve <id>",
    }
    qpath = os.path.join(config.QUEUE_DIR, f"video{vid}_review.json")
    json.dump(entry, open(qpath, "w"), indent=2, ensure_ascii=False)
    db.set_case_status(case_id, "published_pending_review")
    return vid, qpath

def main():
    if len(sys.argv) >= 2 and sys.argv[1] == "list":
        vids = db.list_videos()
        if not vids:
            print("Queue is empty - no videos produced yet.")
            return
        for v in vids:
            print(f"#{v['id']} [{v['status']}] case {v['case_id']} | "
                  f"{v['duration']/60:.1f} min | {v['title']}")
    elif len(sys.argv) >= 3 and sys.argv[1] == "approve":
        vid = int(sys.argv[2])
        db.set_video_status(vid, "approved")
        print(f"Video #{vid} approved - ready for upload when credentials are set.")
    else:
        print(__doc__)

if __name__ == "__main__":
    main()
