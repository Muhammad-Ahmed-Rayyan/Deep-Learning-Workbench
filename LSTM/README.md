# LSTM

## Objective
Binary sentiment classification (positive/negative) of movie reviews using
a Long Short-Term Memory (LSTM) network. This project mirrors the RNN
project exactly (same dataset, preprocessing, vocabulary, and hyperparameters)
with only the recurrent cell swapped, to isolate and measure the effect of
LSTM's gating mechanism on long-sequence performance.

## Dataset
- **Source:** IMDB Reviews (`stanfordnlp/imdb` via HuggingFace `datasets`)
- **Samples:** 25,000 train / 25,000 test (balanced: 50% positive, 50% negative)
- **Preprocessing:** lowercased, HTML tags stripped, non-alphanumeric characters removed, whitespace tokenization
- **Vocabulary:** top 20,000 most frequent tokens (+ `<pad>`, `<unk>`)
- **Sequence length:** truncated/padded to 200 tokens

## Model Architecture
```
Input (token ids, padded)
 -> Embedding(vocab_size=20000, dim=128, padding_idx=0)
 -> LSTM(hidden_dim=128, batch_first=True)   # gated cell: input/forget/output gates + cell state
 -> Dropout(0.3)
 -> Dense(1)  # sigmoid via BCEWithLogitsLoss
```
- Optimizer: Adam (lr=0.001)
- Loss: BCEWithLogitsLoss
- Epochs: 8, Batch size: 64
- Sequences packed with `pack_padded_sequence` to ignore padding in the LSTM computation
- **Only architectural difference from the RNN project:** `nn.RNN` replaced with `nn.LSTM`

## How to Run
```cmd
pip install -r ../requirements.txt
```
1. Run `train_colab.py` in Google Colab (GPU)
2. Download `models/lstm_model.pth`, `models/vocab.json`, and `results/` back into local folder
3. Run `python evaluate.py` locally

## Outputs
- `models/lstm_model.pth`, `models/vocab.json`
- `results/training_curves.png`
- `results/confusion_matrix.png`
- `results/roc_curve.png`
- `results/history.json`, `results/metrics.json`

## Results

| Metric | Value |
|---|---|
| Accuracy | 0.8252 |
| Precision | 0.8600 |
| Recall | 0.7767 |
| F1 Score | 0.8163 |
| ROC-AUC | 0.9037 |

| Class | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| Negative | 0.80 | 0.87 | 0.83 | 12,500 |
| Positive | 0.86 | 0.78 | 0.82 | 12,500 |

## RNN vs LSTM Comparison

Both models were trained on identical data, preprocessing, vocabulary, and
hyperparameters — the only change was the recurrent cell.

| Metric | Vanilla RNN | LSTM | Improvement |
|---|---|---|---|
| Accuracy | 0.7546 | 0.8252 | **+7.06 pts** |
| Precision | 0.7579 | 0.8600 | +10.21 pts |
| Recall | 0.7482 | 0.7767 | +2.85 pts |
| F1 Score | 0.7531 | 0.8163 | +6.32 pts |
| ROC-AUC | 0.8289 | 0.9037 | +7.48 pts |

## Observations
- **LSTM outperforms the vanilla RNN across every metric**, most notably a **+7 point accuracy gain** and **+7.5 point ROC-AUC gain**, confirming the expected behavior: LSTM's input/forget/output gates and separate cell state let it preserve relevant signal across the full 200-token sequence, where the vanilla RNN's hidden state degrades due to vanishing gradients.
- **Precision improved more than recall** (+10.2 vs +2.85 points) — the LSTM became notably better at avoiding false positives on the "Positive" class (precision 0.86) while its recall (0.78) is closer to the RNN's level. This suggests the LSTM built a more confident, discriminative decision boundary rather than just shifting the threshold.
- **Both models remain balanced across classes** (no class imbalance effects, since IMDB is 50/50) — the improvement is purely architectural, not a data artifact.
- **This result directly supports the motivation for LSTM's design**: the gating mechanism specifically targets the long-range dependency problem that limited the RNN, and the ~200-token review length is long enough for that difference to matter.