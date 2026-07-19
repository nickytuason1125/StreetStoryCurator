"""
One-time setup: downloads the permissively-licensed detector weights that
replace ultralytics YOLO (AGPL-3.0), and saves them so the app can load them
fully offline afterward.

  D-FINE-nano (ustc-community/dfine-nano-coco, Apache-2.0)
      -> models/dfine_nano/  (HuggingFace local_files_only=True convention,
         same as ChiaroscuroHead's models/vision_probe/)
  Mask R-CNN (torchvision maskrcnn_resnet50_fpn, BSD-3)
      -> torch hub's own weight cache (~/.cache/torch/hub/checkpoints/);
         this call just warms that cache so the first real app run doesn't
         make a surprise runtime network call.

Run once: python scripts/download_detectors.py
Requires network. Not imported by any runtime code path.
"""
from pathlib import Path

_DEST = Path(__file__).resolve().parent.parent / "models" / "dfine_nano"


def main() -> None:
    print(f"Downloading ustc-community/dfine-nano-coco -> {_DEST}")
    from transformers import AutoModelForObjectDetection, AutoImageProcessor
    model = AutoModelForObjectDetection.from_pretrained("ustc-community/dfine-nano-coco")
    processor = AutoImageProcessor.from_pretrained("ustc-community/dfine-nano-coco")
    _DEST.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(_DEST))
    processor.save_pretrained(str(_DEST))
    print("D-FINE-nano ready for offline use.")

    print("Warming torch hub cache for Mask R-CNN (torchvision, BSD-3)…")
    from torchvision.models.detection import (
        maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights,
    )
    maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.COCO_V1)
    print("Mask R-CNN weights cached — subsequent loads are fully offline.")


if __name__ == "__main__":
    main()
