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
import csv
from collections import defaultdict
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
    min_good: int = MIN_GOOD_MATCHES,
    min_inliers: int = MIN_INLIERS,
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
        if len(good) < min_good:
            continue

        src = np.float32([tpl.keypoints[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([kp_frame[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        if H is None or mask is None:
            continue

        inliers = int(mask.sum())
        if inliers < min_inliers:
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


def write_exposure_report(
    path: Path,
    names: list[str],
    detected_frames: dict[int, int],
    tracked_frames: dict[int, int],
    area_pct_sum: dict[int, float],
    area_pct_n: dict[int, int],
    sec_per_frame: float,
    clip_seconds: float,
) -> None:
    """Write per-logo exposure metrics to CSV.

    `detected_frames` counts processed frames where the logo was actually
    matched. `tracked_frames` additionally counts frames bridged by ByteTrack
    (between a track's first and last sighting), which is the more honest
    on-screen-duration measure since real footage drops detections during
    occlusion/blur. Seconds are scaled by `sec_per_frame = stride / fps` so the
    numbers stay correct even when only every Nth frame is processed.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for class_id, name in enumerate(names):
        det = detected_frames.get(class_id, 0)
        trk = max(tracked_frames.get(class_id, 0), det)
        avg_area = (area_pct_sum.get(class_id, 0.0) / area_pct_n[class_id]
                    if area_pct_n.get(class_id) else 0.0)
        tracked_seconds = trk * sec_per_frame
        rows.append({
            "logo": name,
            "detected_frames": det,
            "detected_seconds": round(det * sec_per_frame, 2),
            "tracked_frames": trk,
            "tracked_seconds": round(tracked_seconds, 2),
            "pct_of_clip": round(100 * tracked_seconds / clip_seconds, 1) if clip_seconds else 0.0,
            "avg_screen_area_pct": round(avg_area, 2),
        })

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Exposure report written to: {path}")
    for r in rows:
        print(f"  {r['logo']:>12}: {r['tracked_seconds']:6.2f}s on screen "
              f"({r['pct_of_clip']:.1f}% of clip), {r['detected_frames']} detected frames, "
              f"avg area {r['avg_screen_area_pct']:.2f}%")


def process_video(args: argparse.Namespace) -> None:
    # Build a detector-agnostic `detect_fn(frame_bgr) -> sv.Detections` so the
    # tracking + exposure code below is shared by feature matching and YOLO.
    if args.detector == "yolo":
        from ultralytics import YOLO  # heavy import; only needed for this path
        print(f"Loading YOLO weights from {args.weights} ...")
        model = YOLO(args.weights)
        names = [model.names[i] for i in sorted(model.names)]

        def detect_fn(frame: np.ndarray) -> sv.Detections:
            res = model.predict(frame, conf=args.conf, iou=0.5,
                                verbose=False, device="cpu")[0]
            return sv.Detections.from_ultralytics(res)
    else:
        detector = build_detector(args.detector)
        matcher = build_matcher(args.detector)
        logo_dir = Path(args.logos)
        print(f"Loading logos from {logo_dir}/ using {args.detector.upper()} ...")
        templates = load_logos(logo_dir, detector, args.detector)
        if not templates:
            raise SystemExit("No usable logo templates found. Add images to the logos folder.")
        names = [t.name for t in templates]

        def detect_fn(frame: np.ndarray) -> sv.Detections:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Detect on a downscaled copy for speed, then map boxes back up.
            if args.scale != 1.0:
                small = cv2.resize(gray, None, fx=args.scale, fy=args.scale,
                                   interpolation=cv2.INTER_AREA)
                det = detect_in_frame(small, templates, detector, matcher,
                                      args.min_good, args.min_inliers)
                if len(det):
                    det.xyxy = det.xyxy / args.scale
                return det
            return detect_in_frame(gray, templates, detector, matcher,
                                   args.min_good, args.min_inliers)

    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    video_info = sv.VideoInfo.from_video_path(args.video)
    print(f"Video: {video_info.width}x{video_info.height} @ {video_info.fps:.1f}fps, "
          f"{video_info.total_frames} frames")

    # ByteTrack gives each logo a stable id across frames and, crucially, lets us
    # count on-screen time through brief detection dropouts. Effective frame rate
    # accounts for --stride so the lost-track buffer spans the right wall-clock.
    eff_fps = max(video_info.fps / max(args.stride, 1), 1.0)
    # Tuned for sparse, high-precision feature-match detections of a single
    # distinctive logo: keep low-confidence hits (we trust the matcher's RANSAC
    # gate), activate immediately, and bridge gaps up to ~2s of dropped frames.
    tracker = sv.ByteTrack(
        frame_rate=round(eff_fps),
        track_activation_threshold=0.1,
        minimum_consecutive_frames=1,
        lost_track_buffer=round(eff_fps * 2),
    ) if args.track else None

    frame_area = float(video_info.width * video_info.height)
    detected_frames: dict[int, int] = defaultdict(int)
    area_pct_sum: dict[int, float] = defaultdict(float)
    area_pct_n: dict[int, int] = defaultdict(int)
    # tracker_id -> [class_id, first_proc_idx, last_proc_idx] for gap bridging
    track_span: dict[int, list[int]] = {}

    frames = sv.get_video_frames_generator(args.video, stride=args.stride)
    hits = 0
    processed = 0
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with sv.VideoSink(str(out_path), video_info=video_info) as sink:
        for i, frame in enumerate(frames):
            if args.max_frames and processed >= args.max_frames:
                break

            detections = detect_fn(frame)

            if tracker is not None:
                detections = tracker.update_with_detections(detections)

            if len(detections):
                hits += 1

            # Accumulate exposure stats for this processed frame.
            for class_id in set(int(c) for c in detections.class_id):
                detected_frames[class_id] += 1
            for xyxy, class_id in zip(detections.xyxy, detections.class_id):
                w = max(xyxy[2] - xyxy[0], 0.0)
                h = max(xyxy[3] - xyxy[1], 0.0)
                area_pct_sum[int(class_id)] += 100.0 * (w * h) / frame_area
                area_pct_n[int(class_id)] += 1
            if detections.tracker_id is not None:
                for class_id, tid in zip(detections.class_id, detections.tracker_id):
                    span = track_span.setdefault(int(tid), [int(class_id), processed, processed])
                    span[2] = processed

            labels = [
                f"{names[c]}{f' #{t}' if t is not None else ''} {conf:.2f}"
                for c, conf, t in zip(
                    detections.class_id,
                    detections.confidence,
                    detections.tracker_id if detections.tracker_id is not None
                    else [None] * len(detections),
                )
            ]
            annotated = box_annotator.annotate(frame.copy(), detections)
            annotated = label_annotator.annotate(annotated, detections, labels=labels)
            sink.write_frame(annotated)
            processed += 1
            if processed % 25 == 0:
                print(f"  ...{processed} frames processed, {hits} with detections")

    print(f"Done. Processed {processed} frames, {hits} contained a logo.")
    print(f"Annotated video written to: {out_path}")

    # Bridge gaps: a logo counts as on-screen for every processed frame between a
    # track's first and last sighting (within ByteTrack's lost-track buffer).
    tracked_present: dict[int, set] = defaultdict(set)
    for class_id, first, last in track_span.values():
        tracked_present[class_id].update(range(first, last + 1))
    tracked_frames = {c: len(p) for c, p in tracked_present.items()}

    sec_per_frame = args.stride / max(video_info.fps, 1.0)
    clip_seconds = video_info.total_frames / max(video_info.fps, 1.0)
    write_exposure_report(Path(args.report), names, detected_frames, tracked_frames,
                          area_pct_sum, area_pct_n, sec_per_frame, clip_seconds)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Detect known logos in a video.")
    p.add_argument("--video", required=True, help="path to input video")
    p.add_argument("--logos", default="logos", help="folder of reference logo images")
    p.add_argument("--output", default="output/annotated.mp4", help="output video path")
    p.add_argument("--report", default="output/exposure.csv",
                   help="per-logo exposure CSV output path")
    p.add_argument("--detector", choices=["orb", "sift", "yolo"], default="orb",
                   help="orb=fast feature match; sift=robust feature match "
                        "(best for wordmarks); yolo=trained detector (best recall)")
    p.add_argument("--weights", default="runs/logo_yolo/weights/best.pt",
                   help="YOLO weights path (used when --detector yolo)")
    p.add_argument("--conf", type=float, default=0.25,
                   help="YOLO confidence threshold (used when --detector yolo)")
    p.add_argument("--stride", type=int, default=1, help="process every Nth frame")
    p.add_argument("--max-frames", type=int, default=0, help="cap frames processed (0=all)")
    p.add_argument("--scale", type=float, default=1.0,
                   help="downscale factor for detection, e.g. 0.5 (faster; output stays full-res)")
    p.add_argument("--min-good", type=int, default=MIN_GOOD_MATCHES,
                   help="min good matches before homography")
    p.add_argument("--min-inliers", type=int, default=MIN_INLIERS,
                   help="min RANSAC inliers to accept a detection (lower=more recall)")
    p.add_argument("--no-track", dest="track", action="store_false",
                   help="disable ByteTrack temporal tracking / gap bridging")
    p.set_defaults(track=True)
    return p.parse_args()


if __name__ == "__main__":
    process_video(parse_args())
