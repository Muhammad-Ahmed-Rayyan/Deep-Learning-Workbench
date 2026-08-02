# Deep Learning Workbench

A collection of deep learning implementations covering core neural network
architectures — from foundational feedforward networks to convolutional,
recurrent, and transfer learning approaches. Each project is self-contained
with its own dataset, training script, evaluation pipeline, and documented
results.

## Projects

| Project | Task | Dataset | Status |
|---|---|---|---|
| [ANN](./ANN) | Binary classification | Wine Quality (UCI) | ✅ Complete |
| [CNN](./CNN) | Image classification | Fashion-MNIST | 🔲 In Progress |
| [RNN](./RNN) | Sentiment analysis | IMDB Reviews | 🔲 Planned |
| [LSTM](./LSTM) | Sentiment analysis | IMDB Reviews | 🔲 Planned |
| [TransferLearning](./TransferLearning) | Image classification | CIFAR-10 | 🔲 Planned |

## Structure

Each project folder follows the same layout:
```
<ProjectName>/
├── data/           # Dataset (downloaded automatically or included)
├── models/         # Saved trained model weights
├── results/         # Metrics, plots, training curves
├── train_colab.py  # Training script (run on GPU via Google Colab)
├── evaluate.py     # Evaluation script (run locally)
└── README.md       # Architecture, dataset details, results, observations
```

## Tech Stack

- **Languages:** Python
- **Frameworks:** TensorFlow / Keras, PyTorch
- **Data & Evaluation:** NumPy, Pandas, scikit-learn, Matplotlib, Seaborn
- **Training environment:** Google Colab (GPU)

## How to Run

### Option A — Local (CPU, for evaluation only — recommended after training in Colab)
```cmd
cd ANN
python -m venv venv
venv\Scripts\activate
pip install -r ../requirements.txt
python evaluate.py
```

### Option B — Google Colab (for training, GPU-backed)
1. Push this repo to GitHub first.
2. In a new Colab notebook:
```python
!git clone https://github.com/<your-username>/<your-repo-name>.git
%cd <your-repo-name>/ANN
!pip install -r ../requirements.txt
!python train_colab.py
```
3. Download the generated `models/ann_model.keras` and `results/test_data.npz` from Colab's file browser.
4. Place them into your local `ANN/models/` and `ANN/results/` folders respectively.

## Results Summary

_(Updated as each project is completed — see individual project READMEs for full details)_

| Project | Key Metric | Score |
|---|---|---|
| ANN | Accuracy / ROC-AUC | 0.90 / 0.88 |