# Flower Classification using CNNs

## Project Description
Transfer learning project using VGG19 and YOLOv5 for 102-category flower classification.

## Requirements
- Python 3.8+
- PyTorch
- TensorFlow/Keras
- See requirements.txt for full list

## Installation
```bash
pip install -r requirements.txt
```

## Usage
```bash
# Prepare data
python src/data_preparation.py

# Train models
python src/train_vgg19.py --split 1
python src/train_yolov5.py --split 1

# Evaluate
python src/evaluate.py
```

## Results
- VGG19 Test Accuracy: XX%
- YOLOv5 Test Accuracy: XX%

## GitHub Repository
[Link to your repo]

## Dataset
- Primary: Oxford 102 Flowers
- Additional: [if you used any]
```
