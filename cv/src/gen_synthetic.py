"""Generate a synthetic YOLO dataset for a single logo from one reference crop.

We have no labelled footage, only a clean logo image. So we *manufacture* labels:
paste the logo onto real backgrounds under random geometric/photometric
augmentation and record the resulting bounding box. Using frames sampled from the
actual clip as backgrounds keeps the domain (lighting, grain, motion blur) close
to inference time.

Output layout (Ultralytics format):

    dataset/
      images/{train,val}/*.jpg
      labels/{train,val}/*.txt    # one "<cls> cx cy w h" line per logo (normalised)
      data.yaml

Limitations (honest): the reference crop is opaque (no alpha) so logos are pasted
as warped quads; the real logo already present in some background frames is left
unlabelled, adding mild label noise. Good enough to learn this logo's appearance
and generalise better than feature matching; swap in hand-labelled data later for
production accuracy.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np


def sample_backgrounds(video: str, count: int, seed: int) -> list[np.ndarray]:
    """Grab `count` evenly-spaced frames from the clip to use as backgrounds."""
    cap = cv2.VideoCapture(video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    idxs = np.linspace(0, max(total - 1, 0), num=count).astype(int)
    frames = []
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if ok:
            frames.append(frame)
    cap.release()
    if not frames:
        raise SystemExit(f"Could not read frames from {video}")
    return frames


def warp_logo(logo: np.ndarray, target_w: int, rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    """Return (warped_bgr, mask) for the logo at a random rotation/perspective.

    The mask marks real logo pixels so we don't paste black corners introduced by
    the warp, and so the bounding box is tight.
    """
    h0, w0 = logo.shape[:2]
    scale = target_w / w0
    logo = cv2.resize(logo, (target_w, max(int(h0 * scale), 1)), interpolation=cv2.INTER_AREA)
    h, w = logo.shape[:2]
    mask = np.full((h, w), 255, np.uint8)

    # Random perspective: jitter each corner by up to ~18% of the size.
    jx, jy = 0.18 * w, 0.18 * h
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = src + np.float32([[rng.uniform(-jx, jx), rng.uniform(-jy, jy)] for _ in range(4)])
    pad_x, pad_y = int(jx) + 2, int(jy) + 2
    dst += [pad_x, pad_y]
    out_w, out_h = w + 2 * pad_x, h + 2 * pad_y
    M = cv2.getPerspectiveTransform(src, dst)
    logo_w = cv2.warpPerspective(logo, M, (out_w, out_h))
    mask_w = cv2.warpPerspective(mask, M, (out_w, out_h))

    # Random in-plane rotation about the centre.
    angle = rng.uniform(-25, 25)
    R = cv2.getRotationMatrix2D((out_w / 2, out_h / 2), angle, 1.0)
    logo_w = cv2.warpAffine(logo_w, R, (out_w, out_h))
    mask_w = cv2.warpAffine(mask_w, R, (out_w, out_h))

    # Photometric jitter (brightness/contrast) + occasional blur.
    alpha, beta = rng.uniform(0.6, 1.3), rng.uniform(-30, 30)
    logo_w = cv2.convertScaleAbs(logo_w, alpha=alpha, beta=beta)
    if rng.random() < 0.5:
        k = rng.choice([3, 5])
        logo_w = cv2.GaussianBlur(logo_w, (k, k), 0)

    _, mask_w = cv2.threshold(mask_w, 127, 255, cv2.THRESH_BINARY)
    return logo_w, mask_w


def paste(bg: np.ndarray, logo_w: np.ndarray, mask_w: np.ndarray, rng: random.Random):
    """Paste warped logo at a random valid position; return YOLO bbox or None."""
    H, W = bg.shape[:2]
    lh, lw = logo_w.shape[:2]
    if lw >= W or lh >= H:
        return None
    x = rng.randint(0, W - lw)
    y = rng.randint(0, H - lh)
    roi = bg[y:y + lh, x:x + lw]
    m3 = mask_w[..., None].astype(bool)
    roi[:] = np.where(m3, logo_w, roi)

    ys, xs = np.where(mask_w > 0)
    if len(xs) == 0:
        return None
    x0, x1, y0, y1 = xs.min() + x, xs.max() + x, ys.min() + y, ys.max() + y
    cx, cy = (x0 + x1) / 2 / W, (y0 + y1) / 2 / H
    bw, bh = (x1 - x0) / W, (y1 - y0) / H
    return cx, cy, bw, bh


def build(args: argparse.Namespace) -> None:
    rng = random.Random(args.seed)
    logo = cv2.imread(args.logo, cv2.IMREAD_COLOR)
    if logo is None:
        raise SystemExit(f"Could not read logo {args.logo}")
    backgrounds = sample_backgrounds(args.video, args.bg_frames, args.seed)
    print(f"Loaded logo {logo.shape[1]}x{logo.shape[0]} and {len(backgrounds)} background frames")

    root = Path(args.out)
    for split in ("train", "val"):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)

    counts = {"train": args.train, "val": args.val}
    for split, n in counts.items():
        made = 0
        for i in range(n):
            bg = backgrounds[rng.randrange(len(backgrounds))].copy()
            H, W = bg.shape[:2]
            lines = []
            if rng.random() >= args.negative_frac:  # else: a hard negative (no logo)
                for _ in range(rng.choice([1, 1, 2])):
                    target_w = int(W * rng.uniform(0.08, 0.35))
                    logo_w, mask_w = warp_logo(logo, max(target_w, 12), rng)
                    box = paste(bg, logo_w, mask_w, rng)
                    if box:
                        lines.append(f"0 {box[0]:.6f} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f}")
            stem = f"{split}_{i:05d}"
            cv2.imwrite(str(root / "images" / split / f"{stem}.jpg"), bg)
            (root / "labels" / split / f"{stem}.txt").write_text("\n".join(lines))
            made += 1
        print(f"  {split}: wrote {made} images")

    yaml = (
        f"path: {root.resolve()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"names:\n  0: {args.name}\n"
    )
    (root / "data.yaml").write_text(yaml)
    print(f"Wrote dataset + data.yaml to {root}/")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Synthesize a YOLO dataset for one logo.")
    p.add_argument("--logo", default="logos/lenovo.png", help="reference logo crop")
    p.add_argument("--video", required=True, help="clip to sample backgrounds from")
    p.add_argument("--out", default="dataset", help="output dataset root")
    p.add_argument("--name", default="lenovo", help="class name")
    p.add_argument("--train", type=int, default=240, help="number of train images")
    p.add_argument("--val", type=int, default=50, help="number of val images")
    p.add_argument("--bg-frames", type=int, default=80, help="background frames to sample")
    p.add_argument("--negative-frac", type=float, default=0.15,
                   help="fraction of images with no logo (hard negatives)")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


if __name__ == "__main__":
    build(parse_args())
