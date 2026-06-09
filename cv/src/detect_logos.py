"""
Detect specific known logos in soccer-game video using classic feature
matching (OpenCV ORB/SIFT + homography), with `supervision` for annotation
and video I/O.

This finds *particular* logos you provide as reference images (e.g. a sponsor
crop) rather than "any logo". For each video frame we match keypoints from the
frame against each reference logo, estimate a homography, and project the
logo's outline into the frame to get a bounding box.

Usage:
    python src/detect_logos.py \
        --video videos/match.mp4 \
        --logos logos/ \
        --output output/annotated.mp4

    # Faster on CPU: process every 3rd frame, cap at 300 frames
    python src/detect_logos.py --video videos/match.mp4 --logos logos/ \
        --stride 3 --max-frames 300

Logos: drop one image per logo into the --logos folder (PNG/JPG). The file
name (without extension) becomes the label, e.g. logos/emirates.png -> "emirates".
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import supervision as sv

# Tunables. Feature matching is a trade-off between recall and false positives;
# these defaults are conservative to avoid spurious boxes.
LOWE_RATIO = 0.75            # ratio test threshold for "good" matches
MIN_GOOD_MATCHES = 12        # minimum good matches before attempting homography
MIN_INLIERS = 10             # minimum RANSAC inliers to accept a detection
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


@dataclass
class LogoTemplate:
    """A reference logo with precomputed keypoints/descriptors."""

    name: str
    keypoints: tuple
    descriptors: np.ndarray
    corners: np.ndarray  # (4,1,2) outline of the reference image


def build_detector(kind: str):
    """Create a feature detector. ORB is fast and free; SIFT is more robust
    but slower (both ship with modern opencv-python)."""
    if kind == "sift":
        return cv2.SIFT_create()
    return cv2.ORB_create(nfeatures=2000)


def build_matcher(kind: str) -> cv2.BFMatcher:
    """Brute-force matcher with the right norm for the detector."""
    norm = cv2.NORM_L2 if kind == "sift" else cv2.NORM_HAMMING
    return cv2.BFMatcher(norm)


def load_logos(folder: Path, detector, kind: str) -> list[LogoTemplate]:
    """Load every image in `folder` as a logo template."""
    templates: list[LogoTemplate] = []
    for path in sorted(folder.iterdir()):
        if path.suffix.lower() not in IMAGE_EXTS:
            continue
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            print(f"  ! skipping unreadable image: {path.name}")
            continue
        kp, des = detector.detectAndCompute(gray, None)
        if des is None or len(kp) < MIN_GOOD_MATCHES:
            print(f"  ! '{path.name}' has too few features ({0 if des is None else len(kp)}); "
                  f"use a larger/sharper crop")
            continue
        h, w = gray.shape[:2]
        corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
        templates.append(LogoTemplate(path.stem, kp, des, corners))
        print(f"  + loaded logo '{path.stem}' ({len(kp)} keypoints)")
    return templates


def detect_in_frame(
    frame_gray: np.ndarray,
    templates: list[LogoTemplate],
    detector,
    matcher: cv2.BFMatcher,
) -> sv.Detections:
    """Return supervision Detections for all logos found in one frame."""
    kp_frame, des_frame = detector.detectAndCompute(frame_gray, None)
    boxes: list[list[float]] = []
    confidences: list[float] = []
    class_ids: list[int] = []

    if des_frame is None or len(kp_frame) == 0:
        return sv.Detections.empty()

    for class_id, tpl in enumerate(templates):
        # k-NN match + Lowe ratio test to keep only confident correspondences.
        knn = matcher.knnMatch(tpl.descriptors, des_frame, k=2)
        good = [m for pair in knn if len(pair) == 2
                for m, n in [pair] if m.distance < LOWE_RATIO * n.distance]
        if len(good) < MIN_GOOD_MATCHES:
            continue

        src = np.float32([tpl.keypoints[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([kp_frame[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        if H is None or mask is None:
            continue

        inliers = int(mask.sum())
        if inliers < MIN_INLIERS:
            continue

        # Project the logo outline into the frame -> axis-aligned bounding box.
        projected = cv2.perspectiveTransform(tpl.corners, H).reshape(-1, 2)
        x1, y1 = projected.min(axis=0)
        x2, y2 = projected.max(axis=0)
        h, w = frame_gray.shape[:2]
        x1, x2 = np.clip([x1, x2], 0, w - 1)
        y1, y2 = np.clip([y1, y2], 0, h - 1)
        if x2 - x1 < 5 or y2 - y1 < 5:
            continue

        boxes.append([float(x1), float(y1), float(x2), float(y2)])
        confidences.append(min(1.0, inliers / max(len(good), 1)))
        class_ids.append(class_id)

    if not boxes:
        return sv.Detections.empty()

    return sv.Detections(
        xyxy=np.array(boxes, dtype=float),
        confidence=np.array(confidences, dtype=float),
        class_id=np.array(class_ids, dtype=int),
    )


def process_video(args: argparse.Namespace) -> None:
    detector = build_detector(args.detector)
    matcher = build_matcher(args.detector)

    logo_dir = Path(args.logos)
    print(f"Loading logos from {logo_dir}/ using {args.detector.upper()} ...")
    templates = load_logos(logo_dir, detector, args.detector)
    if not templates:
        raise SystemExit("No usable logo templates found. Add images to the logos folder.")
    names = [t.name for t in templates]

    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    video_info = sv.VideoInfo.from_video_path(args.video)
    print(f"Video: {video_info.width}x{video_info.height} @ {video_info.fps:.1f}fps, "
          f"{video_info.total_frames} frames")

    frames = sv.get_video_frames_generator(args.video, stride=args.stride)
    hits = 0
    processed = 0
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with sv.VideoSink(str(out_path), video_info=video_info) as sink:
        for i, frame in enumerate(frames):
            if args.max_frames and processed >= args.max_frames:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            detections = detect_in_frame(gray, templates, detector, matcher)
            if len(detections):
                hits += 1
            labels = [
                f"{names[c]} {conf:.2f}"
                for c, conf in zip(detections.class_id, detections.confidence)
            ]
            annotated = box_annotator.annotate(frame.copy(), detections)
            annotated = label_annotator.annotate(annotated, detections, labels=labels)
            sink.write_frame(annotated)
            processed += 1
            if processed % 25 == 0:
                print(f"  ...{processed} frames processed, {hits} with detections")

    print(f"Done. Processed {processed} frames, {hits} contained a logo.")
    print(f"Annotated video written to: {out_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Detect known logos in a video.")
    p.add_argument("--video", required=True, help="path to input video")
    p.add_argument("--logos", default="logos", help="folder of reference logo images")
    p.add_argument("--output", default="output/annotated.mp4", help="output video path")
    p.add_argument("--detector", choices=["orb", "sift"], default="orb",
                   help="feature detector (orb=fast, sift=robust)")
    p.add_argument("--stride", type=int, default=1, help="process every Nth frame")
    p.add_argument("--max-frames", type=int, default=0, help="cap frames processed (0=all)")
    return p.parse_args()


if __name__ == "__main__":
    process_video(parse_args())
