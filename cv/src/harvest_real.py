"""Auto-label REAL frames with the SIFT detector (teacher) for YOLO training.

This is the weak-supervision step: the feature matcher is high-precision but
low-recall, so its hits make trustworthy labels even though it misses many
frames. We train YOLO (student) on these real, correctly-lit, motion-blurred,
fabric-deformed examples so it learns the *actual* logo appearance and — the
whole point — generalises to the frames the teacher missed.

Appends frames + YOLO labels into an existing dataset dir (alongside synthetic
data from gen_synthetic.py), so training sees both. Run gen_synthetic.py first
to create data.yaml.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import supervision as sv

sys.path.insert(0, str(Path(__file__).resolve().parent))
from detect_logos import build_detector, build_matcher, load_logos, detect_in_frame


def harvest(args: argparse.Namespace) -> None:
    detector = build_detector("sift")
    matcher = build_matcher("sift")
    templates = load_logos(Path(args.logos), detector, "sift")
    if not templates:
        raise SystemExit("No logo templates found.")

    root = Path(args.out)
    for split in ("train", "val"):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)

    info = sv.VideoInfo.from_video_path(args.video)
    print(f"Harvesting from {info.total_frames} frames at stride {args.stride}, "
          f"min_inliers={args.min_inliers} ...")
    frames = sv.get_video_frames_generator(args.video, stride=args.stride)
    kept = 0
    for i, frame in enumerate(frames):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        det = detect_in_frame(gray, templates, detector, matcher,
                              args.min_good, args.min_inliers)
        if not len(det):
            continue
        H, W = frame.shape[:2]
        lines = []
        for xyxy, cls in zip(det.xyxy, det.class_id):
            x0, y0, x1, y1 = xyxy
            cx = np.clip((x0 + x1) / 2 / W, 0, 1)
            cy = np.clip((y0 + y1) / 2 / H, 0, 1)
            bw = np.clip((x1 - x0) / W, 0, 1)
            bh = np.clip((y1 - y0) / H, 0, 1)
            lines.append(f"{int(cls)} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        split = "val" if kept % args.val_every == 0 else "train"
        stem = f"real_{i:05d}"
        cv2.imwrite(str(root / "images" / split / f"{stem}.jpg"), frame)
        (root / "labels" / split / f"{stem}.txt").write_text("\n".join(lines))
        kept += 1
        if kept % 20 == 0:
            print(f"  ...{kept} labeled frames so far (scanned {i})")

    print(f"Harvested {kept} real labeled frames into {root}/")
    if kept < 30:
        print("WARNING: few positives — consider lower --min-inliers or --stride 1.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Auto-label real frames with SIFT for YOLO.")
    p.add_argument("--video", required=True)
    p.add_argument("--logos", default="logos")
    p.add_argument("--out", default="dataset")
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--min-good", type=int, default=12)
    p.add_argument("--min-inliers", type=int, default=6)
    p.add_argument("--val-every", type=int, default=7, help="route every Nth frame to val")
    return p.parse_args()


if __name__ == "__main__":
    harvest(parse_args())
