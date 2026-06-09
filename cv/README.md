# Logo Detection in Soccer Video

Detect **specific known logos** (sponsor brands) in soccer-game video and write
an annotated video with bounding boxes + labels. Intended for sponsor-exposure /
brand-visibility analytics.

Detection is **reference-based feature matching** (OpenCV ORB/SIFT + homography)
— no training, no GPU. You supply one cropped reference image per logo;
`supervision` handles annotation and video I/O.

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
└── src/
    ├── detect_logos.py  # main pipeline
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
detections across long gaps. If recall matters, the upgrade path is a trained
YOLO logo detector (`detect_in_frame` is the only piece that swaps out — the
tracking + exposure code is reused). See `HANDOFF.md` §4 and §8.
