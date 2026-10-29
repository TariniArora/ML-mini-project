# Data

We did **not** train any model ourselves. This folder holds only test/demo data.

| Item | What it is | Used for |
|---|---|---|
| Pretrained weights (`../models/`) | YOLOv8n and MobileNetV3-SSDLite, both trained by their authors on **MS-COCO 2017** (80 classes, ~118k training images; we only use the `person` class) | Person detection |
| `samples/test_walk.mp4` | **Synthetic** 22 s video (440 frames, 20 fps) made by `tools/make_test_video.py`: real pedestrian crops from the `bus.jpg` image bundled with the Ultralytics package, pasted on a drawn room background and moved along a scripted path | Automatic testing without a webcam |
| `samples/test_walk_gt.csv` | Ground truth for the video: the pasted target box and distractor box for each frame, plus the scenario phase (A–G) | `tools/evaluate_synthetic.py`, `tools/benchmark.py` |
| `samples/test_frame.jpg` | One frame of the video | Quick image test, `tests/test_pipeline.py` |
| `collected/` (created when you use it) | Your own webcam frames from `collect_data.py` | Optional: your own qualitative test set |

## Regenerate the synthetic video

```
python tools/make_test_video.py
```

## Collect your own frames (optional)

```
python data/collect_data.py --label lab
python data/collect_data.py --label corridor --autolabel   # + YOLO pseudo-labels
```

Pseudo-labels come from the pretrained detector, so they are **not** ground truth.
Check them by hand before you use them to measure accuracy.
