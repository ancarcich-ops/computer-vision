# Handoff — Logo Detection in Soccer Video

Pick this up in a fresh Claude Code session connected to the **new repo
`ancarcich-ops/computer-vision`**. This doc is self-contained: it has the goal,
decisions, environment notes, verification results, full source, and next
steps. The same code is also backed up in `ancarcich-ops/claudecode` under
`cv/` on branch `claude/nice-archimedes-bpv90x`.

---

## 1. Goal

Detect **specific known logos** (sponsor brands) in **soccer-game video**, and
write out an annotated video with boxes + labels. Typical end use:
sponsor-exposure / brand-visibility analytics.

## 2. Decisions already made (don't re-litigate)

| Question | Decision |
|---|---|
| Which logos | **Specific known logos** — you supply reference crops, not "any logo" |
| Input | **Video clips** |
| Detection method | **Feature matching** (OpenCV ORB/SIFT + homography) — no training, no GPU |
| Annotation/video I/O | **`supervision`** (v0.28.0) |
| Project location | **Standalone repo** `ancarcich-ops/computer-vision`. Backup copy in `ancarcich-ops/claudecode` at `cv/`. |
| Language | Python 3.9+ (tested on 3.11). NOT TypeScript — independent of the Next.js app. |

## 3. Environment notes (from the build session)

- Runs **CPU-only** (no GPU was available). Feature matching is fine on CPU;
  full-frame-rate video on long clips is slow — use `--stride` / `--max-frames`.
- Outbound network worked (pypi, github reachable).
- Remote Claude Code containers are **ephemeral** — only what's committed to a
  repo survives. That's why this lives in a repo, not a loose folder.

## 4. Why this method (and its limits)

`supervision` does NOT detect — it annotates/tracks/handles video. Detection is
**reference-based feature matching**: provide a cropped image per logo; for each
frame we match keypoints, estimate a homography (RANSAC), and project the logo
outline into the frame → bounding box.

- **Works well:** distinctive, fairly large, planar logos — pitch-side ad
  boards, big jersey/sleeve sponsors.
- **Struggles:** tiny/blurry logos, motion blur, extreme angles, fabric
  deformation.
- **If recall is poor on real footage:** the upgrade path is a **trained YOLO
  detector** on a logo dataset (e.g. Roboflow Universe has sponsor/logo
  datasets). `supervision` plugs straight into YOLO outputs, so the annotation/
  video code below is reusable — only the per-frame detector swaps out.

## 5. Verification done this session

Built a synthetic demo (feature-rich "ACME" logo composited onto a noisy moving
background) and ran the pipeline:

```
loaded logo 'acme' (260 keypoints)
Video: 640x360 @ 15fps, 30 frames
Done. Processed 30 frames, 23 contained a logo.
```

A sampled output frame showed the ACME logo correctly boxed and labeled
`acme 0.97`. ✅ Pipeline is confirmed end-to-end (demo → match → annotated mp4).

## 6. Project layout

```
cv/                      # (root of the new repo, or cv/ subfolder)
├── requirements.txt
├── README.md
├── HANDOFF.md           # this file
├── logos/               # reference logo crops (input; filename = label)
├── videos/              # input video clips
├── output/              # annotated results (generated)
└── src/
    ├── detect_logos.py  # main pipeline
    └── make_demo.py     # synthetic demo generator (verification)
```

## 7. Setup & run

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# verify with synthetic demo (no footage needed):
python src/make_demo.py
python src/detect_logos.py --video videos/demo.mp4 --logos logos/ --output output/demo_annotated.mp4

# real usage:
#   - drop one crop per logo in logos/  (logos/emirates.png -> label "emirates")
#   - drop your clip in videos/
python src/detect_logos.py --video videos/match.mp4 --logos logos/ \
    --output output/annotated.mp4 --detector orb --stride 2 --max-frames 500
```

`requirements.txt`:
```
supervision==0.28.0
opencv-python>=4.8
numpy>=1.24
```

## 8. Suggested next steps (roadmap)

1. **Get real inputs:** a short real match clip + 2–3 real sponsor logo crops,
   and re-run. This is the real test of the feature-matching approach.
2. **Add temporal tracking:** integrate `sv.ByteTrack` so each logo gets a
   stable ID across frames → enables on-screen *duration* per sponsor (the core
   metric for exposure analytics). Hook lives in the video loop in
   `detect_logos.py`.
3. **Exposure report:** accumulate per-logo frame counts / screen-area-% / total
   seconds, and emit a CSV or JSON summary alongside the video.
4. **If feature matching underperforms** on small/blurry logos: train (or pull a
   pretrained) YOLO logo detector via `ultralytics`/Roboflow `inference`, and
   swap `detect_in_frame()` for the model call — the rest of the pipeline stays.
5. **Speed:** tune `--stride`, cache logo descriptors (already done at load),
   consider FLANN matcher for SIFT on long clips.

## 9. How to wire this into the new session

1. Start a new Claude Code session **on the `ancarcich-ops/computer-vision`
   repo**, and tell Claude: *"Continue the logo detection project — see
   HANDOFF.md."* (The build session was scoped to `claudecode` only and
   couldn't push here directly.)
2. If `computer-vision` is empty, the fastest path is to **also add the
   `ancarcich-ops/claudecode` repo** to that session and copy its `cv/` folder
   (branch `claude/nice-archimedes-bpv90x`) into the new repo root. If you only
   have this Markdown, Claude can recreate the files from Section 10.
3. Run the Section 7 verification to confirm the environment, then start on the
   roadmap.

---

## 10. Full source (for recreation if needed)

### `src/detect_logos.py`
See the committed file — same content as in the `claudecode` backup. Key
functions: `build_detector`/`build_matcher` (ORB vs SIFT), `load_logos`
(precomputes keypoints per reference image), `detect_in_frame` (kNN match +
Lowe ratio test + RANSAC homography → `sv.Detections`), `process_video`
(`sv.get_video_frames_generator` + `sv.VideoSink`, box + label annotators).
Tunables at top: `LOWE_RATIO=0.75`, `MIN_GOOD_MATCHES=12`, `MIN_INLIERS=10`.

### `src/make_demo.py`
Generates `logos/acme.png` (feature-rich synthetic logo) and `videos/demo.mp4`
(logo composited at moving positions/scales over a noisy background) to verify
the pipeline without real footage.

> The complete, runnable versions of both files are in this repo under `src/`.
> If you only have this Markdown file, ask Claude to regenerate them from the
> descriptions above — but normally you'll have the repo.
