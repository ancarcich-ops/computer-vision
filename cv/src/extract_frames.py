"""Extract frames from a clip for manual annotation.

The SIFT teacher can't bootstrap dense/small/repeated logos (e.g. perimeter LED
boards), so those need real labels. This dumps evenly-spaced frames to a folder
you load into a labeling tool (Roboflow / Label Studio / CVAT), box the logos,
and export YOLO format. That export trains directly via train_yolo.py.

    python src/extract_frames.py --video clips/game.mp4 --out frames_to_label --num 200

Tips:
  * --num spreads frames across the whole clip (good scene variety for labeling);
    --stride takes every Nth frame instead.
  * Aim for a few hundred frames covering different angles/board rotations.
  * Label EVERY visible instance in each frame (all boards), or the model learns
    to suppress real logos.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def extract(args: argparse.Namespace) -> None:
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Could not open {args.video}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    if args.num > 0:
        indices = np.linspace(0, max(total - 1, 0), num=args.num).astype(int)
    else:
        indices = np.arange(0, total, max(args.stride, 1))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = Path(args.video).stem
    written = 0
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            continue
        cv2.imwrite(str(out / f"{stem}_{int(idx):06d}.jpg"), frame,
                    [cv2.IMWRITE_JPEG_QUALITY, args.quality])
        written += 1
    cap.release()
    print(f"Wrote {written} frames to {out}/ (clip {total} frames @ {fps:.1f}fps)")
    print("Next: label these in Roboflow/Label Studio, export YOLO format, then "
          "train_yolo.py --data <export>/data.yaml")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Dump frames from a clip for labeling.")
    p.add_argument("--video", required=True)
    p.add_argument("--out", default="frames_to_label")
    p.add_argument("--num", type=int, default=200,
                   help="number of evenly-spaced frames (0 = use --stride)")
    p.add_argument("--stride", type=int, default=15, help="every Nth frame (if --num 0)")
    p.add_argument("--quality", type=int, default=95, help="JPEG quality")
    return p.parse_args()


if __name__ == "__main__":
    extract(parse_args())
