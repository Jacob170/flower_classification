# Flower Classification using CNNs

## Project Description
Transfer learning project using VGG19 and YOLOv5 for 102-category flower classification.

## Dataset
- **Oxford 102 Flowers Dataset**
- 8,189 images across 102 flower categories
- Split: 50% training / 25% validation / 25% testing
- Two random seeds (42 and 123) used to evaluate model robustness

## Results

| Model | Seed 42 | Seed 123 | Trainable Params |
|-------|---------|----------|------------------|
| VGG19 | ~80% | ~80% | ~124M |
| YOLOv5-cls | 91.75% | 91.89% | ~130K |

## Requirements
- Python 3.8+
- PyTorch 2.8.0
- matplotlib
-sickcit learn
- See requirements.txt for full list


## Installation
### Please Use VENV!

### 1. Clone the repository
```bash
git clone https://github.com/Jacob170/flower_classification.git
cd flower_classification
```

### 2. Create virtual environment (recommended)

```bash
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows
```


```bash
pip install -r requirements.txt
```

## Code Stracture 

1. data_preparation.py - splits the dats 25%|25%|50%
2. evaluate - prints the plots of accuracy and loss 
3. train_vgg19.py - training class for vgg
4. train_yolov5.py - training class for yolo
5. preproccessing - augmentation normalization of iamges before training 
6. main - main program runs everything (except the data_preparation.py so no split)


## Usage

### Step 1: Prepare data (run once)
```bash
python src/data_preparation.py
```
This creates stratified train/val/test splits for seeds 42 and 123.

### Step 2: Train and evaluate
```bash
python src/main.py
```
This runs preprocessing, training, and evaluation for both models.

### Note !! Important !! If you are running this on CPU (without GPU)
in the main change the DataLoader arguments  
to work with num_workers=1 or 2
and the   pin_memory=False  or just remove it 
This code optimized to work with cuda 


## Model Architectures

### VGG19
- **Frozen:** 16 convolutional layers (feature extractor)
- **Trainable:** 3 fully-connected classifier layers
- **Modified:** Final layer changed from 1000 → 102 classes
- **Optimizer:** SGD (lr=0.01, momentum=0.9)
- **Epochs:** 35

### YOLOv5-cls
- **Frozen:** Backbone layers 0-8
- **Trainable:** Classification head (layer 9)
- **Modified:** Final linear layer changed from 1000 → 102 classes
- **Optimizer:** Adam (lr=0.001)
- **Epochs:** 20

## Key Findings

1. **Modern architectures win:** YOLOv5-cls (2020) outperformed VGG19 (2014) by ~12% accuracy
2. **Efficiency matters:** YOLOv5-cls trained 1000x fewer parameters
3. **Both models are robust:** < 0.2% variance between different random seeds
4. **Transfer learning works:** Achieved 92% accuracy with only ~4,000 training images

## Results
- VGG19 Test Accuracy: XX%
- YOLOv5 Test Accuracy: 91.89%

## Authors
Jacob Yaacubov &     Barak Milshtein

## GitHub Repository
https://github.com/Jacob170/flower_classification

## Dataset
- Primary: Oxford 102 Flowers
- Additional: [if you used any]
```
