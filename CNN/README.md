# CNN

## Objective
Multi-class image classification of clothing items using a Convolutional
Neural Network.

## Dataset
- **Source:** Fashion-MNIST (via `torchvision.datasets`, auto-downloaded)
- **Samples:** 60,000 train / 10,000 test
- **Image size:** 28x28 grayscale
- **Classes (10):** T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot

## Preprocessing
- Normalization using Fashion-MNIST channel mean/std (0.2860, 0.3530)
- No augmentation applied (baseline CNN)

## Model Architecture
```
Input (1x28x28)
 -> Conv2d(32, 3x3) -> ReLU -> BatchNorm -> MaxPool(2x2)   # -> 32x14x14
 -> Conv2d(64, 3x3) -> ReLU -> BatchNorm -> MaxPool(2x2)   # -> 64x7x7
 -> Flatten -> Dense(128) -> ReLU -> Dropout(0.4)
 -> Dense(10)
```
- Optimizer: Adam (lr=0.001)
- Loss: CrossEntropyLoss
- Epochs: 15, Batch size: 64

## How to Run
```cmd
pip install -r ../requirements.txt
cd cnn
python train.py
python evaluate.py
```

## Outputs
- `models/cnn_model.pth`
- `results/training_curves.png`
- `results/confusion_matrix.png`
- `results/history.json`
- `results/metrics.json`

## Results

| Metric | Value |
|---|---|
| Accuracy | 0.9246 |
| Precision (macro) | 0.9248 |
| Recall (macro) | 0.9246 |
| F1 (macro) | 0.9246 |

**Per-class F1-score**

| Class | Precision | Recall | F1-score |
|---|---|---|---|
| T-shirt/top | 0.86 | 0.90 | 0.88 |
| Trouser | 0.99 | 0.99 | 0.99 |
| Pullover | 0.91 | 0.86 | 0.88 |
| Dress | 0.93 | 0.92 | 0.92 |
| Coat | 0.88 | 0.89 | 0.88 |
| Sandal | 0.99 | 0.98 | 0.98 |
| **Shirt** | **0.78** | **0.78** | **0.78** |
| Sneaker | 0.97 | 0.97 | 0.97 |
| Bag | 0.98 | 0.99 | 0.98 |
| Ankle boot | 0.97 | 0.97 | 0.97 |

## Observations
- **Overall performance is strong (92.5% accuracy)** for a simple 2-conv-layer baseline, with balanced macro precision/recall — no single class is being systematically ignored.
- **Shirt is the clear weak point (F1 = 0.78)**, well below every other class. This is a well-documented Fashion-MNIST failure mode: at 28x28 grayscale resolution, "Shirt" overlaps visually with **T-shirt/top, Pullover, and Coat** — similar silhouette, no color/texture information to help distinguish them. See `results/confusion_matrix.png` for exact misclassification patterns between these classes.
- **Footwear and accessory classes (Trouser, Sandal, Sneaker, Bag, Ankle boot) all scored 0.97+ F1** — these have more distinctive shapes and are much easier for the model to separate from clothing tops.
- **Next steps to improve Shirt classification:** data augmentation (random rotation/crop), a deeper architecture with more feature maps, or training with class-specific weighting to force more attention on the confusable classes.
- BatchNorm and Dropout(0.4) kept training stable; no signs of overfitting in `results/training_curves.png` (train/val accuracy track closely).