# Optional Models Integration

This document explains how to integrate optional models for enhanced grading in Street Story Curator.

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