# Shared craft constants (compiled once, injected with EVERY niche playbook)

Caption craft (Netflix Timed Text style guide, via kmilo93sd/agent-video-editor):
- <= 42 chars/line, <= 2 lines, reading speed <= 20 chars/s, min duration
  ~0.83 s; break lines at punctuation, never split article-from-noun or a name.
- Close a cue on sentence end, or at 6 s max, or when a speech gap >= 0.6 s
  opens (cut-as-code chunking law). Gaps < 0.3 s between cues get closed.
- Never double-deliver: footage that burns captions must not also ship an
  external .srt with the same text (viewer sees both).

Silence/dead-air cuts (browser-use/video-use measured thresholds):
- Gaps >= 400 ms are the clean cut targets; 150-400 ms usable only with a
  frame check; < 150 ms is mid-phrase — never cut.
- Pad every cut edge 30-200 ms; speaker-handoff air 400-600 ms. ASR word
  ends run 50-150 ms early — extend the cut-out until the room tone is quiet
  (~6 dB over floor) before you cut (konradre two-band edge law: place
  edges from SOUND, never from word timestamps alone).

Punch-ins (social-media-skills + kmilo93sd):
- Scale 1.10-1.20x, never > 1.25x on this genre; switch at cuts that remove
  >= 1 s, else only after 4 s of the same framing; never across a cut;
  space them 5-7 s apart. (Matches the tool law: `timeline.set_zoom` 1.0-1.25.)

Audio (YouTube delivery, verified chain):
- Target -14 LUFS integrated, true peak <= -1 dBTP, two-pass loudnorm, verify
  with ebur128. Chain: gain -> highpass 80 Hz -> afftdn (skip if room noise
  < -60 dB) -> acompressor 3:1 from -24 dB (5 ms attack / 120 ms release)
  -> limiter -2 dB -> one clean gain to target (social-media-skills).
- Music beds sit 18-25 dB under dialogue; SFX 12-24 dB under; narration blocks
  TTS clips must never clip -1 dBTP (kmilo93sd sourced numbers).

Provenance lawbook for evidence-footage editing (0xsline/OpenChatCut
news-rough-cut):
- ONLY the sources' raw sound may enter the cut. NO music beds, NO added VO
  over evidence audio, NO SFX on evidence moments — the genre's credibility
  IS the provenance. (Narration blocks ride muted, darkened b-roll — never
  over a speaker's evidence audio.)
- Speech is cut only at sentence ends / natural pauses / speaker turns;
  never splice two contexts into a fake causality; never present conjecture
  as fact; restrained dissolves only, no filters or stickers.
- The verified-outcome framing law overrides this only where the dossier
  carries a verdict (see Legal & tone precision in the niche playbook).
