# Logo Detection in Soccer Video

Detect **specific known logos** (sponsor brands) in soccer-game video and write
an annotated video with bounding boxes + labels. Intended for sponsor-exposure /
brand-visibility analytics.

Two detection backends share one tracking + exposure pipeline:

1. **Reference feature matching** (OpenCV ORB/SIFT + homography) — no training,
   no GPU, just one cropped reference image per logo. High precision, partial
   recall.
2. **Trained YOLOv8 detector** (`--detector yolo`) — best recall, generalises to
   scale/rotation/occlusion. Trained here from a *single* logo crop via synthetic
   data, so it still needs no hand-labelling. See **[YOLO route](#yolo-route-best-recall)**.

`supervision` handles ByteTrack tracking, annotation, and video I/O.

See [`HANDOFF.md`](HANDOFF.md) for full context: decisions, environment notes,
verification results, and the roadmap.

## Layout

```
cv/
├── requirements.txt
├── README.md
├── HANDOFF.md
├── logos/               # reference logo crops (input; filename = label)
├── videos/              # input video clips
├── output/              # annotated results (generated)
├── dataset/            # synthetic YOLO dataset (generated)
├── runs/               # YOLO training outputs / weights (generated)
└── src/
    ├── detect_logos.py  # main pipeline (orb / sift / yolo backends)
    ├── build_dataset.py # combine many clips + negatives into one YOLO dataset
    ├── harvest_real.py  # SIFT teacher auto-labels real frames (one clip)
    ├── gen_synthetic.py # build a YOLO dataset from one logo crop
    ├── train_yolo.py    # train YOLOv8 on the dataset
    └── make_demo.py     # synthetic demo generator (verification)
```

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Verify (no footage needed)

```bash
python src/make_demo.py
python src/detect_logos.py --video videos/demo.mp4 --logos logos/ \
    --output output/demo_annotated.mp4
```

## Real usage

Drop one crop per logo into `logos/` (`logos/emirates.png` → label `emirates`)
and your clip into `videos/`, then:

```bash
python src/detect_logos.py --video videos/match.mp4 --logos logos/ \
    --output output/annotated.mp4 --report output/exposure.csv \
    --detector sift --min-inliers 6 --stride 2
```

This writes an annotated video **and** a per-logo exposure CSV.

### Detector choice (important)

`--detector sift` is strongly recommended for **text / wordmark logos**
(sponsor patches like "Lenovo"). The default `orb` is fast but produces too few
matches on flat text-on-fabric logos — in testing it found zero where SIFT found
the logo reliably. Use `orb` only for busy, highly-textured logos where speed
matters.

### Tracking + exposure report

`sv.ByteTrack` is on by default: it gives each logo a stable id across frames
and bridges short detection dropouts (occlusion, motion blur, turns), so the
on-screen *duration* isn't undercounted every time a single frame misses. The
CSV (`--report`) contains per logo:

| column | meaning |
|---|---|
| `detected_frames` | processed frames the matcher actually hit |
| `detected_seconds` | that count converted to seconds (stride-aware) |
| `tracked_frames` / `tracked_seconds` | including frames bridged by ByteTrack |
| `pct_of_clip` | tracked seconds as a share of the whole clip |
| `avg_screen_area_pct` | mean bounding-box area as % of frame |

Disable tracking with `--no-track` (then tracked == detected).

### Flags & tunables

- `--detector {orb,sift}` — see above.
- `--min-inliers N` (default 10) — lower → more recall, more false positives.
- `--min-good N` (default 12) — min good matches before homography.
- `--scale F` (e.g. `0.5`) — downscale frames for detection (~3× faster), at a
  recall cost; output video stays full-res. Best left at `1.0` for small logos.
- `--stride N` / `--max-frames N` — keep long clips fast on CPU.

### Known limit (recall)

Feature matching is **high-precision, partial-recall** on small/deformable
logos: it catches clear, head-on instances and misses angled/occluded/tiny ones,
so exposure numbers *undercount*. ByteTrack mitigates this but can't invent
detections across long gaps. The fix is the trained YOLO backend below.

## YOLO route (best recall)

A trained detector generalises to scale, rotation, and partial occlusion far
better than feature matching — but normally needs a hand-labelled dataset. We
avoid manual labelling with **weak supervision**: the SIFT detector acts as a
*teacher* that auto-labels real frames, and YOLO is the *student* trained on
them. The student learns the logo's real appearance and generalises past the
teacher's recall.

> ⚠️ **The #1 lesson: train on MANY clips, not one.** A model trained on a
> single clip learns the *scene* (e.g. "centred person's torso = logo"), not the
> logo — it scored great in-domain but fired on ~60% of an unseen clip's frames
> where no logo existed. Diverse sources + hard negatives are what make it
> generalise. Use `build_dataset.py` below, not the single-clip path.

### Recommended: multi-clip dataset

```bash
# clips/      -> several clips that CONTAIN the logo (different games/angles)
# negatives/  -> logo-FREE clips, ideally with people/jerseys (suppress
#                torso/background false positives). Optional but important.

# 1. Build one combined dataset: SIFT auto-labels positives, negatives become
#    empty-label backgrounds. Hold a whole clip out for honest evaluation.
python src/build_dataset.py --clips clips/ --negatives negatives/ \
    --out dataset --val-clips held_out_game --fresh

# 2. Train (base weights auto-download once)
python src/train_yolo.py --data dataset/data.yaml --epochs 40 --imgsz 512 \
    --name logo_real

# 3. Test on a clip that was NOT in training — the real generalisation check
python src/detect_logos.py --video videos/unseen.mp4 --detector yolo \
    --weights runs/logo_real/weights/best.pt --conf 0.25 \
    --output output/annotated.mp4 --report output/exposure.csv
```

`build_dataset.py` flags: `--stride` (positive sampling), `--neg-stride`
(negative sampling), `--min-inliers` (teacher precision gate), `--synthetic N`
(add pasted-logo composites for variety), `--val-clips` (hold whole clips out of
training). Multi-logo works out of the box: drop one crop per logo in `logos/`
(filename = class name) — every script reads all of them.

Only `detect_fn` differs between the orb/sift/yolo backends; ByteTrack and the
exposure report are shared, so the [tracking + exposure](#tracking--exposure-report)
section applies identically. Extra deps (`ultralytics`, `torch`) are in
`requirements.txt`.

### Single-clip helpers (bootstrapping / quick tests)

`gen_synthetic.py` (paste one crop onto frames) and `harvest_real.py` (SIFT
auto-label one clip) are the building blocks `build_dataset.py` wraps. Useful in
isolation, but **synthetic-only doesn't transfer** (0.99 synth mAP, ~0 real
recall — it overfits the paste artifacts) and **single-clip training memorises
the scene**. Always prefer the multi-clip path for anything real.

### Why confidence matters

`--conf` is the precision/recall dial. Lower it (e.g. `0.10`) for more recall,
raise it to cut false positives. Tune it against a held-out clip, not the
training clips.
