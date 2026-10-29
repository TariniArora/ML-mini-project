# Pretrained models

| File | Model | Source / training data | Licence |
|---|---|---|---|
| `yolov8n.pt` | Ultralytics YOLOv8-nano object detector (about 3.2 M parameters) | Released by Ultralytics, trained on MS-COCO 2017 | AGPL-3.0 |
| `ssdlite320_mobilenet_v3_large_coco-a79551df.pth` | torchvision SSDLite320 with a MobileNetV3-Large backbone | Released by PyTorch/torchvision, trained on MS-COCO 2017 | BSD-3-Clause |

Both files are used **as released**. We did no fine-tuning.
If a file is missing, it downloads on first run:

* YOLOv8n is downloaded automatically by `ultralytics`.
* SSDLite is downloaded from download.pytorch.org.
