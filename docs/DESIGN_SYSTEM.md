# DirectorOS Studio — Design System

How the Studio UI is designed, and why. The competitive reference set is
**Adobe Premiere Pro, DaVinci Resolve, CapCut Desktop, and Descript**; the
audit of our previous UI found it read as a developer dashboard, not an NLE.
This system fixes that with the patterns those four products agree on.

## Principles

1. **The app is a viewport, not a page.** Full-screen shell, no document
   scroll. Panels dock; only panel *contents* scroll.
2. **Three surface tiers, one hairline.** `bg-0` shell, `bg-1` panels,
   `bg-2` raised (headers, inputs, track headers), hairline `line-1`
   dividers. No fourth gray, no glassmorphism.
3. **One accent.** Electric blue `#4d9fff` is reserved for the playhead,
   selection, active states, and the primary action. Red is destructive
   only. Green is success only. When everything is accented, nothing is.
4. **Timecodes are monospace tabular.** `HH:MM:SS:FF` at the media frame
   rate; the hero timecode rides the top bar and the transport bar.
5. **Real controls, no native widgets.** Custom toggles, sliders, buttons
   (4 kinds: primary / ghost / danger / icon). Native checkboxes and
   `<video controls>` are banned.
6. **Empty states teach.** Every empty state names what will appear, when,
   and offers the CTA that fills it.

## Layout (Studio project page)

```
┌───────────────────────────────────────────────────────────────────┐
│ top bar 48px: logo · case · status chip · stage+progress · actions│
├────┬──────────────┬────────────────────────────┬──────────────────┤
│rail│ left panel   │ program monitor            │ pipeline         │
│44px│ 300px        │ (letterboxed black + blur  │ (per-step bars)  │
│    │ media/src/   │  canvas + transport 44px)  ├──────────────────┤
│    │ caps/tools/  │                            │ sandbox console  │
│    │ publish      │                            │  (terminal feed) │
├────┴──────────────┴────────────────────────────┴──────────────────┤
│ timeline dock 268px: toolbar · ruler · 5 tracks (V2 V1 A1 CC FX)  │
└───────────────────────────────────────────────────────────────────┘
```

Same skeleton as Resolve's Edit page (media pool / monitor / inspector /
timeline) — editors already know where to look.

## Timeline anatomy

- **Ruler** — adaptive major/minor ticks (`rulerTicks()`): labels on major
  only, `HH:MM:SS:FF` at high zoom, `mm:ss` below; click/drag seeks.
- **Track headers** — name chip + eye (hide) + mute + lock per track type,
  exactly like Premiere. Eye/lock actually work.
- **Clips** — 4px radius, per-type palette (`--clip-*`); video clips carry a
  **real filmstrip** tiled from the ingest scrubsheet sprite (one frame per
  2 s of source, `scrubBackground()`); selected clips get a 2px accent
  outline and drag handles for in/out trims.
- **Captions** — amber chips with the cue text; **Effects** — rose blocks
  with a hover delete.
- **Playhead** — 2px accent line spanning ruler + tracks with a flag handle
  in the ruler (Premiere's shape).
- **Toolbar** — version chip, split at playhead, snap magnet, zoom slider
  + fit; selected-clip info on the right.

## Color tokens (`globals.css`)

| Token | Value | Use |
| --- | --- | --- |
| `--bg-0/1/2/3` | `#131417 / #1a1b1f / #222329 / #2b2c33` | shell / panels / raised / hover |
| `--line-1/2` | `#282a30 / #3a3c45` | hairlines / strong borders |
| `--tx-1/2/3` | `#e9ebef / #a4aab5 / #6d7480` | text tiers |
| `--acc` | `#4d9fff` | playhead, selection, active, primary |
| `--ok / --warn / --err` | `#3fd68f / #f5b544 / #f16d6d` | status only |
| `--clip-video/audio/card/cc/fx` | blue / green / purple / amber / rose | NLE track semantics |

## Typography

System UI sans (Inter-class) at 10–13px; micro-labels are 10px/600/uppercased
/tracked. All timecodes: `ui-monospace` stack, tabular numerals, 19px hero.

## Settings

Provider cards with a status dot (last probe result, persisted locally),
grid fields, and a Test button that tests **what will be used** — typed
values when present, otherwise the saved vault key (probe resolution:
`params > vault > env`, so saved keys never need re-entry).
