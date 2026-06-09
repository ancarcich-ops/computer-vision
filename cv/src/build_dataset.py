"""Build one combined YOLO dataset from MANY clips — the fix for the
single-clip shortcut problem.

A detector trained on one clip learns the scene ("centred person's torso"),
not the logo. Feeding it diverse sources forces the logo to be the only
consistent signal. This orchestrates three ingredients into one dataset:

  * POSITIVES  (--clips DIR): every video is SIFT-auto-labelled (teacher).
  * NEGATIVES  (--negatives DIR, optional): frames from logo-FREE footage
    (ideally with people/jerseys) become empty-label backgrounds, teaching
    "person != logo". This is the direct antidote to torso false positives.
  * SYNTHETIC  (--synthetic N, optional): logo crops pasted onto sampled
    backgrounds for extra scale/position variety.

Honest evaluation: keep at least one clip OUT of --clips entirely and test on
it (run detect_logos.py --detector yolo). Per-frame val on the training clips
measures memorisation, not generalisation. Use --val-clips to hold whole
clips out of training for a within-dataset generalisation check.

Workflow:
    python src/build_dataset.py --clips clips/ --negatives negatives/ --out dataset
    python src/train_yolo.py --data dataset/data.yaml --epochs 40 --imgsz 512
    python src/detect_logos.py --video held_out.mp4 --detector yolo \
        --weights runs/logo_real/weights/best.pt
"""
from __future__ import annotations

import argparse
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import supervision as sv

sys.path.insert(0, str(Path(__file__).resolve().parent))
from detect_logos import build_detector, build_matcher, load_logos
from harvest_real import iter_labeled_frames
import gen_synthetic as gs

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}


def video_files(folder: str) -> list[Path]:
    d = Path(folder)
    if not d.is_dir():
        raise SystemExit(f"{folder} is not a directory")
    return sorted(p for p in d.iterdir() if p.suffix.lower() in VIDEO_EXTS)


def choose_split(clip_name: str, val_clips: list[str], val_frac: float,
                 rng: random.Random) -> str:
    """Route a frame to train/val. If --val-clips is given, those clips go
    entirely to val and everything else to train; otherwise split per-frame."""
    if val_clips:
        return "val" if any(v in clip_name for v in val_clips) else "train"
    return "val" if rng.random() < val_frac else "train"


def write_example(root: Path, split: str, stem: str, frame, lines: list[str]) -> None:
    cv2.imwrite(str(root / "images" / split / f"{stem}.jpg"), frame)
    (root / "labels" / split / f"{stem}.txt").write_text("\n".join(lines))


def build(args: argparse.Namespace) -> None:
    rng = random.Random(args.seed)
    detector = build_detector("sift")
    matcher = build_matcher("sift")
    logos_dir = Path(args.logos)
    templates = load_logos(logos_dir, detector, "sift")
    if not templates:
        raise SystemExit("No logo templates found.")
    names = [t.name for t in templates]
    val_clips = [v.strip() for v in args.val_clips.split(",") if v.strip()]

    root = Path(args.out)
    if root.exists() and args.fresh:
        shutil.rmtree(root)
    for split in ("train", "val"):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)

    stats: dict[str, list[int]] = defaultdict(lambda: [0, 0])  # source -> [train, val]

    # --- POSITIVES: SIFT-auto-labelled real frames from every clip ----------
    positives = video_files(args.clips)
    if not positives:
        raise SystemExit(f"No videos found in {args.clips}")
    print(f"Positives: {len(positives)} clip(s)")
    for clip in positives:
        n = 0
        for idx, frame, lines in iter_labeled_frames(
                clip, templates, detector, matcher,
                args.stride, args.min_good, args.min_inliers):
            split = choose_split(clip.name, val_clips, args.val_frac, rng)
            write_example(root, split, f"{clip.stem}_{idx:06d}", frame, lines)
            stats[clip.name][0 if split == "train" else 1] += 1
            n += 1
        print(f"  {clip.name}: {n} labelled frames")

    # --- NEGATIVES: empty-label backgrounds from logo-free footage ----------
    if args.negatives:
        negs = video_files(args.negatives)
        print(f"Negatives: {len(negs)} clip(s) @ stride {args.neg_stride}")
        for clip in negs:
            n = 0
            for idx, frame in enumerate(
                    sv.get_video_frames_generator(str(clip), stride=args.neg_stride)):
                split = choose_split(clip.name, val_clips, args.val_frac, rng)
                write_example(root, split, f"neg_{clip.stem}_{idx:06d}", frame, [])
                stats[f"NEG:{clip.name}"][0 if split == "train" else 1] += 1
                n += 1
            print(f"  {clip.name}: {n} negative frames")

    # --- SYNTHETIC: pasted-logo composites for scale/position variety -------
    if args.synthetic > 0:
        logo_imgs = []
        for name in names:
            hit = next((p for p in sorted(logos_dir.iterdir()) if p.stem == name), None)
            logo_imgs.append(cv2.imread(str(hit)) if hit else None)
        backgrounds = []
        for clip in positives:
            for f in sv.get_video_frames_generator(str(clip), stride=max(args.stride * 8, 8)):
                backgrounds.append(f)
                if len(backgrounds) >= 120:
                    break
        if backgrounds and any(im is not None for im in logo_imgs):
            print(f"Synthetic: {args.synthetic} composites")
            for k in range(args.synthetic):
                bg = backgrounds[rng.randrange(len(backgrounds))].copy()
                H, W = bg.shape[:2]
                cls = rng.randrange(len(names))
                logo = logo_imgs[cls]
                lines = []
                if logo is not None:
                    tw = max(int(W * rng.uniform(0.05, 0.3)), 12)
                    logo_w, mask_w = gs.warp_logo(logo, tw, rng)
                    box = gs.paste(bg, logo_w, mask_w, rng)
                    if box:
                        lines.append(f"{cls} {box[0]:.6f} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f}")
                split = "val" if rng.random() < args.val_frac else "train"
                write_example(root, split, f"synth_{k:06d}", bg, lines)
                stats["SYNTHETIC"][0 if split == "train" else 1] += 1

    # --- data.yaml + summary ------------------------------------------------
    names_block = "\n".join(f"  {i}: {n}" for i, n in enumerate(names))
    (root / "data.yaml").write_text(
        f"path: {root.resolve()}\ntrain: images/train\nval: images/val\n"
        f"names:\n{names_block}\n")

    tot_tr = sum(v[0] for v in stats.values())
    tot_va = sum(v[1] for v in stats.values())
    print(f"\nDataset at {root}/ — classes={names}")
    for src, (tr, va) in stats.items():
        print(f"  {src:>28}: train={tr:4d} val={va:3d}")
    print(f"  {'TOTAL':>28}: train={tot_tr:4d} val={tot_va:3d}")
    if tot_tr < 200:
        print("NOTE: small dataset — add more clips for better generalisation.")
    if not args.negatives:
        print("NOTE: no --negatives given. Add logo-free clips to suppress "
              "torso/background false positives.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build a combined multi-clip YOLO dataset.")
    p.add_argument("--clips", required=True, help="folder of positive clips (contain the logo)")
    p.add_argument("--negatives", default="", help="folder of logo-free clips (hard negatives)")
    p.add_argument("--logos", default="logos", help="folder of reference logo crops")
    p.add_argument("--out", default="dataset", help="output dataset root")
    p.add_argument("--stride", type=int, default=2, help="frame stride for positive clips")
    p.add_argument("--neg-stride", type=int, default=15, help="frame stride for negative clips")
    p.add_argument("--min-good", type=int, default=12)
    p.add_argument("--min-inliers", type=int, default=6, help="SIFT teacher precision gate")
    p.add_argument("--synthetic", type=int, default=0, help="number of synthetic composites to add")
    p.add_argument("--val-frac", type=float, default=0.15, help="per-frame val fraction")
    p.add_argument("--val-clips", default="", help="comma-separated clip-name substrings to "
                   "hold entirely in val (overrides per-frame split)")
    p.add_argument("--fresh", action="store_true", help="wipe --out before building")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


if __name__ == "__main__":
    build(parse_args())
