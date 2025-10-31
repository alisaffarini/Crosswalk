# Crosswalk Detection

A lightweight crosswalk detection model for semantic segmentation tasks. This project serves as a specialized component for a larger semantic segmentation pipeline inspired by PaddleSeg.

## Overview

This model detects crosswalks in street-level imagery and outputs bounding box coordinates along with a binary classification. Built on a modified MobileNetV3-inspired architecture (LYTNetV2), it provides efficient inference suitable for real-time applications.

## Architecture

- **Backbone**: LYTNetV2 (MobileNetV3-based)
- **Outputs**: 
  - Binary crosswalk classification (is/is not crosswalk)
  - Normalized bounding box coordinates [x1, y1, x2, y2]

## Files

- `model.py` - Main model wrapper with dual-head architecture
- `LYTNetV2.py` - Lightweight backbone network architecture
- `CV EVAL.py` - Inference and visualization script
- `moreefficientweights.pth` - Pretrained model weights

## Usage

```python
from model import MakeModel
import torch

# Load model
model = MakeModel(pretrained=True)
model.eval()

# Run inference
output = model(input_tensor)
coordinates = output['coordinates']
is_crosswalk = output['IScrosswalk']
```

## Requirements

- PyTorch
- torchvision
- OpenCV
- Pillow
- NumPy

## Context

This model is designed as a specialized detection module within a broader semantic segmentation framework, providing focused crosswalk identification capabilities that complement general-purpose scene understanding tasks.

