# Labeling logos for the YOLO detector

For logos the SIFT teacher can't auto-label — small, repeated, motion-blurred
perimeter LED boards, on-screen graphics, etc. — you label a few hundred frames
once, and the trained model handles the rest. This is the reliable, production
path for those cases.

## 1. Extract frames

```bash
python src/extract_frames.py --video clips/game_hd.mp4 --out frames_to_label --num 250
```

`--num` spreads frames across the whole clip for variety. Aim for **200–400
frames** covering different camera angles and board rotations. (I can zip and
send you the folder, then you upload it to a labeling tool.)

## 2. Label in a tool

Use any of these (all free to start, all export YOLO):

- **Roboflow** (easiest, web): create project → upload frames → draw boxes →
  Generate → Export as **YOLOv8** → download the zip.
- **Label Studio** or **CVAT** (self-host/web): same idea, export YOLO format.

Rules that matter:

- **Class name = the logo** (e.g. `lenovo`). Keep it identical across clips so
  classes merge.
- **Box EVERY visible instance** in each frame (all the boards), even small or
  partly occluded ones. Missed instances teach the model to *suppress* real
  logos — that's how you get false negatives.
- A tight box around the wordmark is fine; consistency matters more than pixels.

## 3. Drop the export in and train

A YOLO export looks like:

```
export/
  data.yaml
  train/images/*.jpg   train/labels/*.txt
  valid/images/*.jpg   valid/labels/*.txt
```

Train directly on it:

```bash
python src/train_yolo.py --data export/data.yaml --epochs 50 --imgsz 640 --name logo_board
```

(Use `--imgsz 640` or higher — board logos are small, so resolution helps.)

## 4. Test on a HELD-OUT clip

Always evaluate on footage that wasn't labeled/trained on:

```bash
python src/detect_logos.py --video clips/unseen.mp4 --detector yolo \
    --weights runs/logo_board/weights/best.pt --conf 0.25 \
    --output output/boards.mp4 --report output/boards.csv
```

`--conf` trades recall vs false positives — tune it on the held-out clip.

## Combining with auto-labeled data

Manual board labels and SIFT-auto-labeled jersey data can live in one dataset:
keep the same class names and merge the `images/` + `labels/` folders (or point
`build_dataset.py` at the auto-label clips and copy the manual export's files
into the same `dataset/` tree). One model can then cover multiple renditions of
the same sponsor — which is the end goal: recognize each logo across jerseys,
boards, and graphics.
