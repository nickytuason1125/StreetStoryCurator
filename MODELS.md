# Optional Models Integration

This document explains how to integrate optional models for enhanced grading in FirstCut.

## YuNet Face Detection Model

The YuNet model provides improved face detection capabilities compared to the default Haar cascades.

### Installation

The YuNet model is automatically downloaded when you run the application if it's not present. Alternatively, you can manually download it using:

```bash
python -c "import urllib.request; import os; os.makedirs('models', exist_ok=True); urllib.request.urlretrieve('https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx', 'models/face_detection_yunet_2023mar.onnx')"
```

### Benefits

- Better detection of profiles and partial faces
- Improved performance in low-light conditions
- More accurate face counting for human presence scoring

## NIMA Neural Image Assessment

The NIMA (Neural Image Assessment) model provides enhanced aesthetic scoring based on human ratings.

### Installation

To install the NIMA model:

1. Ensure you have PyTorch installed:
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
   ```

2. Run the NIMA setup script:
   ```bash
   python nima_setup.py
   ```

### Benefits

- Enhanced aesthetic scoring based on 250k human photo ratings
- Improved correlation with professional photography standards
- Better differentiation between technically correct and aesthetically pleasing images

## Model Files Location

All models are stored in the `models/` directory:

- `models/face_detection_yunet_2023mar.onnx` - YuNet face detection model
- `models/onnx/nima.onnx` - NIMA aesthetic assessment model
- `models/onnx/dinov2_small.onnx` - DINOv2 composition analysis (required)
- `models/onnx/mobilevit_aesthetic.onnx` - MobileViT aesthetic proxy (required)

## Usage

Once installed, the models are automatically used by the application. The analyzer will:

1. Use YuNet for face detection when available (falls back to Haar cascades)
2. Apply NIMA scoring as an additional signal in the final grade calculation
3. Continue to work normally even if optional models are not present

## Troubleshooting

If you encounter issues with model integration:

1. Ensure all dependencies are installed:
   ```bash
   pip install torch torchvision onnxscript
   ```

2. Check that the model files exist in the correct locations
3. Restart the application after adding new models

The application is designed to gracefully handle missing optional models and will continue to function with reduced but still effective capabilities.
## Story Mode Vision Models (vision_story_mode.py)

The slot gatekeeper is chosen by measurement, not fashion. The A/B tournament
(`_ab_tournament.json`, 2026-06-14, batch `test_batch_25 classic_street`)
kept **Qwen2.5-VL** as the judge:

- `smolvlm2_22b` was rejected: mean 0.852 with std 0.003 -- it gives every
  photograph the same score (no discriminative signal).
- `qwen3_4b` was rejected: 10.9 s/img for std 0.076 despite Spearman 0.817.
- `internvl3.5_2b` was rejected: 15.7 s/img, Spearman 0.477.

Do not swap the model without rerunning the tournament against the same batch.

Pipeline geometry note (2026-09-21 audit): candidate images are letterboxed
(aspect preserved) into the evaluation canvas and every bounding box the gate
returns is mapped back to true-image coordinates before any spatial fact is
derived. Grayscale/square-stretch preprocessing was removed -- it corrupted
h_gap / subject-area facts and destroyed the tone signal.

## Judge-Verdict Model + Protocol (2026-09-21, `_judge_ab_report_2026.md`)

Production default (config.json): `"JUDGE_MODEL": "phi4-mini:latest"` with
`"JUDGE_MODE": "structured"`. The structured protocol emits schema-constrained
JSON whose claims must cite fact IDs with EXACT packet numbers -- grounding is
programmatic, the word cap is enforced by assembly, temperature 0 + seed.
Measured under this protocol: phi4-mini 0.926 grounding [0.824-0.971], 8.0 s,
0 FAILs; deepseek-r1:8b-q3km 0.963 [0.875-0.990] but lower self-consistency and
1.8x latency (overlap in CIs -- statistically tied; r1 is the accuracy-leaning
alternative). qwen3.5 (4b AND 9b) cannot comply with schema-constrained judging
(100% schema failure) -- family-level exclusion. Fallback chain: structured ->
prose -> deterministic default text. Rerun `_judge_ab2.py` before any change.
