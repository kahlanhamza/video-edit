"""Daily pipeline orchestrator. Resumable at stage AND segment granularity.

  python3 -m pipeline.run_pipeline --daily
"""
import os, sys, json
from . import config, db, stage1_source, stage2_script, stage3_voice
from . import stage4_editor, stage5_package, stage6_publish, gates

def _state_path(case_id):
    return os.path.join(config.CACHE_DIR, f"case{case_id}_pipeline_state.json")

def _load_state(case_id):
    p = _state_path(case_id)
    return json.load(open(p)) if os.path.exists(p) else {}

def _save_state(case_id, st):
    json.dump(st, open(_state_path(case_id), "w"), indent=2)

def produce_case(case_id):
    st = _load_state(case_id)
    print(f"=== producing case #{case_id} ===")
    if not st.get("sourcing"):
        stage1_source.run(case_id)
        st["sourcing"] = True; _save_state(case_id, st)
    else:
        print("[STAGE 1] done")
    if not st.get("anchors"):
        from . import analyze
        analyze.run()
        st["anchors"] = True; _save_state(case_id, st)
    else:
        print("[ANCHORS] done")
    if not st.get("script"):
        anchors_by_clip = {}
        for item in config.USED_POOL:
            ap = os.path.join(config.CACHE_DIR, f"anchors_{item['key']}.json")
            anchors_by_clip[item["key"]] = json.load(open(ap))["anchors"]
        script = stage2_script.run(case_id, anchors_by_clip, target_sec=960)
        st["script"] = True; _save_state(case_id, st)
    else:
        script = json.load(open(os.path.join(config.PACKAGE_DIR,
                    f"case{case_id}_script.json")))
        print("[STAGE 2] done")
    if not st.get("voice"):
        manifest = stage3_voice.run(case_id, script)
        st["voice"] = True; _save_state(case_id, st)
    else:
        manifest = json.load(open(os.path.join(config.AUDIO_DIR,
                    f"case{case_id}_audio_manifest.json")))
        print("[STAGE 3] done")
    if not st.get("render"):
        video_path, duration, timeline, chapters = stage4_editor.run(
            case_id, script, manifest)
        # record lane: keep the timeline in the committed record too
        rec = os.path.join(config.DATA_DIR, "records",
                           f"case{case_id}_timeline.json")
        json.dump({"timeline": timeline, "chapters": chapters},
                  open(rec, "w"), indent=2, ensure_ascii=False)
        st["render"] = True; st["duration"] = duration; _save_state(case_id, st)
    else:
        video_path = os.path.join(config.VIDEO_DIR, f"case{case_id}_final.mp4")
        duration = st["duration"]
        rec = os.path.join(config.DATA_DIR, "records",
                           f"case{case_id}_timeline.json")
        tl = json.load(open(rec))
        timeline, chapters = tl["timeline"], tl["chapters"]
        print("[STAGE 4] done")
    if not st.get("package"):
        thumb, metadata = stage5_package.run(case_id, script, chapters,
                                             timeline, video_path)
        st["package"] = True; _save_state(case_id, st)
    else:
        metadata = json.load(open(os.path.join(config.PACKAGE_DIR,
                    f"case{case_id}_metadata.json")))
        thumb = os.path.join(config.THUMB_DIR, f"case{case_id}_thumb.png")
        print("[STAGE 5] done")
    print("=== running GATES (non-bypassable) ===")
    gates.run(case_id)
    if not st.get("queued"):
        vid, qpath = stage6_publish.queue_for_review(
            case_id, metadata["title"], video_path, thumb, duration, metadata)
        st["queued"] = vid; _save_state(case_id, st)
        print(f"  queued as video #{vid} -> {qpath}")
    else:
        print(f"[STAGE 6] already queued as video #{st['queued']}")
    print(f"=== case #{case_id} COMPLETE: {metadata['title']} "
          f"({duration/60:.1f} min) ===")
    return True

def main():
    if "--daily" in sys.argv:
        case_id = stage1_source.run()
        ok = produce_case(case_id)
        sys.exit(0 if ok else 1)
    else:
        print(__doc__)

if __name__ == "__main__":
    main()
