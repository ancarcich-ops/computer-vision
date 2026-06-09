"""Train a YOLOv8 logo detector on the synthetic dataset (CPU-friendly).

Thin wrapper around Ultralytics so the run is reproducible and the knobs that
matter on a CPU box (model size, imgsz, epochs) are exposed. The base weights
(yolov8n.pt) are fetched once from the Ultralytics GitHub release on first use.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def main(args: argparse.Namespace) -> None:
    data = Path(args.data)
    if not data.exists():
        raise SystemExit(f"{data} not found - run gen_synthetic.py first")

    model = YOLO(args.base)
    model.train(
        data=str(data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device="cpu",
        project=args.project,
        name=args.name,
        exist_ok=True,
        patience=args.patience,
        verbose=True,
    )
    best = Path(args.project) / args.name / "weights" / "best.pt"
    print(f"Best weights: {best}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train YOLOv8 logo detector (CPU).")
    p.add_argument("--data", default="dataset/data.yaml")
    p.add_argument("--base", default="yolov8n.pt", help="base weights to fine-tune")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--imgsz", type=int, default=512)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--patience", type=int, default=12, help="early-stop patience")
    p.add_argument("--project", default="runs")
    p.add_argument("--name", default="logo_yolo")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
