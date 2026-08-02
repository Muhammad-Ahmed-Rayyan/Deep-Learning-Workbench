<div align="center">

# 🧠 Deep Learning Workbench

*A collection of core neural network architectures — from feedforward to convolutional, recurrent, and transfer learning approaches*

![Last Commit](https://img.shields.io/github/last-commit/Muhammad-Ahmed-Rayyan/Deep-Learning-Workbench)
![languages](https://img.shields.io/github/languages/count/Muhammad-Ahmed-Rayyan/Deep-Learning-Workbench)

<br>

Built with the tools and technologies:  
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)

</div>

---

## 🧠 Project Summary

**Deep Learning Workbench** is a collection of deep learning implementations covering core neural network architectures — from foundational feedforward networks to convolutional, recurrent, and transfer learning approaches. Each project is self-contained with its own dataset, training script, evaluation pipeline, and documented results.

RNN and LSTM use identical data, preprocessing, and hyperparameters — only the recurrent cell differs — to isolate the effect of LSTM's gating mechanism. See [`LSTM/README.md`](./LSTM/README.md#rnn-vs-lstm-comparison) for the full comparison.

---

## 🚀 Features

- 🔢 **ANN — Binary Classification**
  Wine quality prediction (good/not good) on the [Wine Quality (UCI)](https://archive.ics.uci.edu/ml/datasets/wine+quality) dataset (1,599 samples, 11 features) using a Feedforward NN (64-32-16-1) with Dropout — **90.0% accuracy, 0.88 ROC-AUC**

- 🖼️ **CNN — Multi-Class Image Classification**
  10 clothing categories on [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) (60k train / 10k test, 28x28 grayscale) using a 2-block CNN (Conv-BatchNorm-MaxPool) in PyTorch — **92.5% accuracy, 0.925 macro F1**

- 💬 **RNN — Binary Sentiment Classification**
  Movie review sentiment on [IMDB Reviews](https://huggingface.co/datasets/stanfordnlp/imdb) (25k train / 25k test) using a Vanilla RNN (Embedding + RNN cell) — **75.5% accuracy, 0.829 ROC-AUC**

- 🔁 **LSTM — Binary Sentiment Classification**
  Same IMDB data as RNN for direct comparison, using an LSTM (Embedding + LSTM cell, gated) — **82.5% accuracy, 0.904 ROC-AUC (+7 pts over RNN)**

- 🏗️ **Transfer Learning — Multi-Class Image Classification**
  10 object categories on [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) (50k train / 10k test, 32x32 color) using a pretrained ResNet18 (ImageNet weights) with frozen backbone + new classifier head — **81.0% accuracy, 0.810 macro F1**

Each project's README contains full architecture diagrams, per-class metrics, confusion matrices, and detailed observations.

---

## 🗃️ Project Structure

```bash
Deep-Learning-Workbench/
├── ANN/
│   ├── data/
│   ├── models/
│   │   └── ann_model.keras
│   │── results/
│   │   ├── confusion_matrix.png
│   │   ├── metrics.json
│   │   ├── roc_curve.png
│   │   ├── test_data.npz
│   │   └── training_curves.png
│   ├── evaluate.py
│   ├── train_colab.py
│   └── README.md
├── CNN/
│   ├── data/
│   ├── models/
│   │── results/
│   │   ├── confusion_matrix.png
│   │   ├── history.json
│   │   ├── metrics.json
│   │   └── training_curves.png
│   ├── evaluate.py
│   ├── train_colab.py
│   └── README.md
├── LSTM/
│   ├── data/
│   ├── models/
│   │   ├── lstm_model.pth
│   │   └── vocab.json
│   │── results/
│   │   ├── confusion_matrix.png
│   │   ├── history.json
│   │   ├── metrics.json
│   │   ├── roc_curve.png
│   │   └── training_curves.png
│   ├── evaluate.py
│   ├── train_colab.py
│   └── README.md
├── RNN/
│   ├── data/
│   ├── models/
│   │   ├── rnn_model.pth
│   │   └── vocab.json
│   │── results/
│   │   ├── confusion_matrix.png
│   │   ├── history.json
│   │   ├── metrics.json
│   │   ├── roc_curve.png
│   │   └── training_curves.png
│   ├── evaluate.py
│   ├── train_colab.py
│   └── README.md
│── TransferLearning/
│   ├── data/
│   ├── models/
│   │   └── resnet18_cifar10.pth
│   │── results/
│   │   ├── confusion_matrix.png
│   │   ├── history.json
│   │   ├── metrics.json
│   │   └── training_curves.png
│   ├── evaluate.py
│   ├── train_colab.py
│   └── README.md
├── .gitignore
├── README.md
└── requirements.txt

```


---

## 🔧 Setup & Installation

Each project (`ANN`, `CNN`, `RNN`, `LSTM`, `TransferLearning`) follows the same two-step workflow: train on GPU via Colab, then evaluate locally.

### Backend — Local Evaluation (CPU, recommended after training in Colab)

```bash
cd <ProjectName>

python -m venv venv

venv\Scripts\activate

pip install -r ../requirements.txt

python evaluate.py
```

### Frontend — Google Colab Training (GPU-backed)

1. Push this repo to GitHub first.
2. In a new Colab notebook:

```python
!git clone https://github.com/Muhammad-Ahmed-Rayyan/Deep-Learning-Workbench.git

%cd Deep-Learning-Workbench/<ProjectName>

!pip install -r ../requirements.txt

!python train_colab.py
```

3. Download the generated files from `models/` and `results/` using Colab's file browser.
4. Place them into your local `<ProjectName>/models/` and `<ProjectName>/results/` folders respectively.

Replace `<ProjectName>` with the project you're working on (e.g. `CNN`, `RNN`).

---

## 📊 Results Summary

| Project | Key Metric | Score |
|---|---|---|
| ANN | Accuracy / ROC-AUC | 0.90 / 0.88 |
| CNN | Accuracy / Macro F1 | 0.925 / 0.925 |
| RNN | Accuracy / ROC-AUC | 0.755 / 0.829 |
| LSTM | Accuracy / ROC-AUC | 0.825 / 0.904 |
| TransferLearning | Accuracy / Macro F1 | 0.810 / 0.810 |

---

<div align="center">

⭐ Found this project useful? Drop a star on GitHub!

</div>