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
    --output output/annotated.mp4 --detector orb --stride 2 --max-frames 500
```

Tunables (top of `src/detect_logos.py`): `LOWE_RATIO`, `MIN_GOOD_MATCHES`,
`MIN_INLIERS`. `--detector sift` is more robust but slower than the default
`orb`; `--stride` / `--max-frames` keep long clips fast on CPU.
