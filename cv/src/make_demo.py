"""
Generate a synthetic logo + a synthetic video that composites that logo onto
a moving background. Used to verify the detection pipeline end-to-end without
real footage.

    python src/make_demo.py
    python src/detect_logos.py --video videos/demo.mp4 --logos logos/ \
        --output output/demo_annotated.mp4

Produces:
    logos/acme.png      reference logo
    videos/demo.mp4     30-frame clip with the logo moving/scaling
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def make_logo(path: Path, size: int = 200) -> np.ndarray:
    """Draw a feature-rich synthetic logo (shapes + text give ORB keypoints)."""
    img = np.full((size, size, 3), 255, dtype=np.uint8)
    cv2.rectangle(img, (10, 10), (size - 10, size - 10), (180, 40, 40), 8)
    cv2.circle(img, (size // 2, size // 2), 55, (40, 40, 200), -1)
    cv2.rectangle(img, (40, 60), (160, 140), (40, 180, 40), 4)
    for x in range(30, size - 30, 18):  # stripes add texture/corners
        cv2.line(img, (x, 30), (x, size - 30), (20, 20, 20), 2)
    cv2.putText(img, "ACME", (35, size // 2 + 12),
                cv2.FONT_HERSHEY_DUPLEX, 1.6, (255, 255, 255), 3)
    cv2.imwrite(str(path), img)
    return img


def make_video(path: Path, logo: np.ndarray, n_frames: int = 30,
               w: int = 640, h: int = 360, fps: int = 15) -> None:
    """Composite the logo at moving positions/scales over a textured background."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
    rng = np.random.default_rng(0)
    # Static textured background so the frame itself has distractor features.
    background = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
    background = cv2.GaussianBlur(background, (7, 7), 0)

    for i in range(n_frames):
        frame = background.copy()
        scale = 0.6 + 0.3 * np.sin(i / 6.0)
        lw = int(logo.shape[1] * scale)
        lh = int(logo.shape[0] * scale)
        resized = cv2.resize(logo, (lw, lh))
        x = int((w - lw) * (0.5 + 0.4 * np.sin(i / 8.0)))
        y = int((h - lh) * (0.5 + 0.4 * np.cos(i / 7.0)))
        x = max(0, min(x, w - lw))
        y = max(0, min(y, h - lh))
        frame[y:y + lh, x:x + lw] = resized
        writer.write(frame)
    writer.release()


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    logo_path = root / "logos" / "acme.png"
    video_path = root / "videos" / "demo.mp4"
    logo_path.parent.mkdir(parents=True, exist_ok=True)
    video_path.parent.mkdir(parents=True, exist_ok=True)

    logo = make_logo(logo_path)
    make_video(video_path, logo)
    print(f"Wrote {logo_path}")
    print(f"Wrote {video_path}")


if __name__ == "__main__":
    main()
