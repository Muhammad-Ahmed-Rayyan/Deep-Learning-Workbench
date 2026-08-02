# 01 - Artificial Neural Network (ANN)

## Objective
Binary classification of red wine quality (Good vs Not Good) using a fully-connected
feedforward neural network, based on physicochemical properties.

## Dataset
- **Source:** UCI Machine Learning Repository - Wine Quality (Red Wine)
- **URL:** https://archive.ics.uci.edu/ml/datasets/wine+quality
- **Samples:** 1,599
- **Features:** 11 physicochemical inputs (acidity, sugar, chlorides, sulfur dioxide, density, pH, sulphates, alcohol, etc.)
- **Target:** Wine quality score (0-10), converted to binary label: `1` if quality >= 7 ("Good"), else `0`

## Preprocessing
- Train/test split: 80/20, stratified on label
- Feature scaling: `StandardScaler` (zero mean, unit variance)

## Model Architecture
```
Input (11 features)
 -> Dense(64, ReLU) -> Dropout(0.3)
 -> Dense(32, ReLU) -> Dropout(0.2)
 -> Dense(16, ReLU)
 -> Dense(1, Sigmoid)
```
- Optimizer: Adam (lr=0.001)
- Loss: Binary Crossentropy
- Regularization: Dropout layers to reduce overfitting
- Early stopping on validation loss (patience=10)

## How to Run
```cmd
pip install -r ../requirements.txt
python train.py
python evaluate.py
```

## Outputs
- `models/ann_model.keras` - trained model
- `results/training_curves.png` - accuracy/loss over epochs
- `results/confusion_matrix.png`
- `results/roc_curve.png`
- `results/metrics.json` - accuracy, precision, recall, F1, ROC-AUC

## Results

| Metric | Value |
|---|---|
| Accuracy | 0.9000 |
| Precision | 0.7619 |
| Recall | 0.3721 |
| F1 Score | 0.5000 |
| ROC-AUC | 0.8809 |

**Classification Report**

| Class | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| 0 (Not Good) | 0.91 | 0.98 | 0.94 | 277 |
| 1 (Good) | 0.76 | 0.37 | 0.50 | 43 |

## Observations
- **Class imbalance drives the metrics.** The test set has 277 "Not Good" wines vs only 43 "Good" wines (~87/13 split). This is why the headline accuracy (90%) looks strong but is misleading on its own.
- **Recall on the minority class (Good wine) is the real weak point (0.37).** The model correctly identifies "Not Good" wines almost perfectly (recall 0.98) but misses roughly 6 out of 10 actual "Good" wines. In a real use case (e.g. quality control), this matters more than overall accuracy.
- **ROC-AUC (0.88) suggests the model has learned meaningful separation** between the classes even though the default 0.5 decision threshold is not well-tuned for the minority class.
- **Next steps to improve recall on class 1:** class weighting (`class_weight` in `model.fit`), oversampling the minority class (e.g. SMOTE), or lowering the classification threshold below 0.5 and re-evaluating precision/recall trade-off.
- Early stopping and dropout controlled overfitting reasonably well; training/validation curves (`results/training_curves.png`) show no major divergence between train and val loss.