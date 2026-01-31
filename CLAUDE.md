# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Transfer learning project for 102-category flower classification using the Oxford 102 Flowers dataset. Implements two architectures:
- **VGG19**: PyTorch/torchvision implementation with frozen feature extraction layers
- **YOLOv5**: Ultralytics YOLOv5 classification model (not detection)

## Data Architecture

### Dataset Structure
- Raw data: `data/raw/` containing JPG images and `imagelabels.mat` (MATLAB format)
- Labels are 1-indexed (1-102) in the .mat file, converted to 0-indexed (0-101) in code
- Stratified splits stored in `data/splits/split_{seed}/` with three subsets:
  - `train/` (50% of data)
  - `val/` (25% of data)
  - `test/` (25% of data)
- Images organized in class folders: `class_000/` through `class_101/`

### Split Seeds
Two random seeds are used for cross-validation:
- `split_42` (seed=42)
- `split_123` (seed=123)

## Commands

### Environment Setup
```bash
# Activate virtual environment (if exists)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Data Preparation
```bash
# Create stratified train/val/test splits from raw data
python src/data_preparation.py
```

### Training Models
```bash
# Run all combinations of models and splits
python src/main.py

# Or train individually (legacy approach mentioned in README):
python src/train_vgg19.py --split 1
python src/train_yolov5.py --split 1
```

Note: The actual training scripts don't currently accept `--split` arguments. The `main.py` orchestrator loops through all model/split combinations.

### Evaluation
Evaluation happens automatically during training via `main.py`. The `evaluate.py` module provides plotting functions called after each training run.

## Model Architectures

### VGG19 (train_vgg19.py)
- Loads `torchvision.models.vgg19(pretrained=True)`
- **Frozen layers**: All `backbone.features` parameters
- **Trainable head**: Replaced `classifier[6]` with:
  - Linear(4096 → 512) → ReLU → Dropout(0.3) → Linear(512 → 102)
- Loss: CrossEntropyLoss (no LogSoftmax in model)
- Optimizer: Adam with lr=0.00001
- Input size: 224×224

### YOLOv5 (train_yolov5.py)
- Loads classification model via `torch.hub.load("ultralytics/yolov5", "custom", path="yolov5s-cls.pt")`
- **Frozen layers**: All parameters except those with "head" in name
- **Trainable head**: Replaced `model.model.head.fc` for 102 classes
- Loss: CrossEntropyLoss
- Optimizer: Adam with lr=0.001
- Input size: 640×640

## Preprocessing (preprocessing.py)

Both models use ImageNet normalization: `mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]`

**Training transforms**: Resize → RandomHorizontalFlip → RandomRotation(10°) → ToTensor → Normalize
**Validation/test transforms**: Resize → ToTensor → Normalize (no augmentation)

## Training Configuration

Common hyperparameters (defined in train_*.py):
- `BATCH_SIZE = 32`
- `EPOCHS = 20`
- `NUM_CLASSES = 102`
- Device: CUDA if available, else CPU

Both model classes implement:
- `train_model(train_loader, val_loader)`: Returns history dict with train/val loss/accuracy
- `test_model(test_loader)`: Returns test accuracy and loss

## Key Implementation Details

- **YOLOv5 dependency**: Requires `yolov5s-cls.pt` pretrained weights in root directory and cloned yolov5 repo
- **Data loading**: Uses `torchvision.datasets.ImageFolder` which expects class subdirectories
- **Evaluation outputs**: Matplotlib plots showing loss and accuracy curves with test performance as horizontal lines
- **No model saving**: Current implementation doesn't persist trained models to disk
- **No command-line arguments**: `main.py` hardcodes the models and splits to iterate over

## File Organization

- `src/data_preparation.py`: Creates stratified splits from raw data
- `src/preprocessing.py`: Transform functions for both architectures
- `src/train_vgg19.py`: VGG19 model class and training logic
- `src/train_yolov5.py`: YOLOv5 model class and training logic
- `src/evaluate.py`: Plotting utilities for loss and accuracy visualization
- `src/main.py`: Orchestrator that trains all model/split combinations
- `models/`: Directory for storing trained model checkpoints (currently unused)
- `results/`: Directory for storing outputs
- `yolov5/`: Cloned ultralytics/yolov5 repository
