# TransferLearning

## Objective
Multi-class image classification on CIFAR-10 using transfer learning —
reusing a ResNet18 pretrained on ImageNet instead of training a CNN from
scratch, to demonstrate how learned visual features generalize to a new task.

## Dataset
- **Source:** CIFAR-10 (via `torchvision.datasets`, auto-downloaded)
- **Samples:** 50,000 train / 10,000 test
- **Image size:** 32x32 color (resized to 224x224 to match ResNet18's expected input)
- **Classes (10):** airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

## Preprocessing
- Resize 32x32 -> 224x224
- Random horizontal flip (train only)
- Normalization using ImageNet mean/std ([0.485, 0.456, 0.406] / [0.229, 0.224, 0.225]) — required since the backbone's learned filters expect this input distribution

## Transfer Learning Approach
- **Base model:** ResNet18 pretrained on ImageNet (`torchvision.models.ResNet18_Weights.IMAGENET1K_V1`)
- **Strategy: Feature extraction (frozen backbone).** All pretrained convolutional layers are frozen (`requires_grad = False`); only the final fully-connected layer is replaced (1000 -> 10 classes) and trained.
- **Why frozen instead of full fine-tuning:** CIFAR-10 (50k images) is small relative to ResNet18's ~11M parameters. Fine-tuning the full network risks overfitting and would take significantly longer to train. Freezing the backbone treats it as a fixed, general-purpose visual feature extractor and only adapts the final decision layer to CIFAR-10's 10 classes — a standard and efficient transfer learning strategy for smaller datasets.
- Optimizer only updates the new `fc` layer's parameters (`filter(lambda p: p.requires_grad, model.parameters())`)

## Model Architecture
```
Input (3x224x224)
 -> ResNet18 backbone (frozen, ImageNet-pretrained)
    conv layers -> residual blocks -> global avg pool
 -> Dense(512 -> 10)   # newly added, trainable
```
- Optimizer: Adam (lr=0.0005)
- Loss: CrossEntropyLoss
- Epochs: 10, Batch size: 64

## How to Run
```cmd
pip install -r ../requirements.txt
```
1. Run `train_colab.py` in Google Colab (GPU)
2. Download `models/resnet18_cifar10.pth` and `results/` back into local folder
   (or run `evaluate.py` directly in Colab if local CPU inference is too slow — see note below)
3. Run `python evaluate.py` (locally or in Colab)

> **Note:** `evaluate.py` resizes all 10,000 test images to 224x224 and runs them through the full ResNet18 — this can be slow on CPU. Running it in Colab (GPU) is recommended if local evaluation takes too long.

## Outputs
- `models/resnet18_cifar10.pth`
- `results/training_curves.png`
- `results/confusion_matrix.png`
- `results/history.json`, `results/metrics.json`

## Results

| Metric | Value |
|---|---|
| Accuracy | 0.8098 |
| Precision (macro) | 0.8104 |
| Recall (macro) | 0.8098 |
| F1 (macro) | 0.8095 |

**Per-class F1-score**

| Class | Precision | Recall | F1-score |
|---|---|---|---|
| airplane | 0.78 | 0.86 | 0.82 |
| automobile | 0.87 | 0.90 | 0.88 |
| bird | 0.79 | 0.75 | 0.77 |
| **cat** | **0.70** | **0.69** | **0.69** |
| deer | 0.76 | 0.78 | 0.77 |
| dog | 0.78 | 0.74 | 0.76 |
| frog | 0.86 | 0.83 | 0.84 |
| horse | 0.82 | 0.83 | 0.83 |
| ship | 0.84 | 0.88 | 0.86 |
| truck | 0.91 | 0.84 | 0.87 |

## Observations
- **81% accuracy with a frozen backbone and only ~5,130 trainable parameters** (the new `fc` layer: 512 x 10 + bias) is a strong demonstration of transfer learning's value — the ResNet18 backbone never saw a CIFAR-10 image during its original ImageNet training, yet its learned low/mid-level features (edges, textures, shapes) transferred well enough that training a single linear layer reached 81% accuracy.
- **Cat is the weakest class (F1 = 0.69)**, consistent with well-known CIFAR-10 difficulty: cats and dogs share similar body shapes, fur textures, and poses at low resolution, making them the hardest pair to separate — see `dog` also scoring below average (F1 = 0.76). See `results/confusion_matrix.png` to confirm the exact cat/dog confusion pattern.
- **Vehicle/vessel classes (automobile, ship, truck) scored highest (F1 0.86-0.88)** — these have more distinctive shapes and consistent backgrounds compared to animal classes, which vary more in pose and appearance.
- **Comparison to CNN-from-scratch (Fashion-MNIST project):** while not a direct apples-to-apples comparison (different dataset, different task difficulty), it's worth noting CIFAR-10 with color images and more visually similar classes is a harder classification problem than Fashion-MNIST, yet transfer learning still reached comparable performance (81% vs CNN's 92.5%) — with far fewer trainable parameters and less training time, since only the final layer was optimized.
- **Next steps to improve accuracy further:** unfreeze and fine-tune the last few ResNet18 layers (not just the classifier head) with a small learning rate, or apply stronger data augmentation (random crop, color jitter) to help the model generalize better on visually similar classes like cat/dog.