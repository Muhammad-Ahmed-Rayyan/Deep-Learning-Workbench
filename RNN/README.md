# RNN

## Objective
Binary sentiment classification (positive/negative) of movie reviews using
a vanilla Recurrent Neural Network (RNN).

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
 -> RNN(hidden_dim=128, batch_first=True)     # vanilla RNN cell
 -> Dropout(0.3)
 -> Dense(1)  # sigmoid via BCEWithLogitsLoss
```
- Optimizer: Adam (lr=0.001)
- Loss: BCEWithLogitsLoss
- Epochs: 8, Batch size: 64
- Sequences packed with `pack_padded_sequence` to ignore padding in the RNN computation

## How to Run
```cmd
pip install -r ../requirements.txt
```
1. Run `train_colab.py` in Google Colab (GPU)
2. Download `models/rnn_model.pth`, `models/vocab.json`, and `results/` back into local folder
3. Run `python evaluate.py` locally

## Outputs
- `models/rnn_model.pth`, `models/vocab.json`
- `results/training_curves.png`
- `results/confusion_matrix.png`
- `results/roc_curve.png`
- `results/history.json`, `results/metrics.json`

## Results

| Metric | Value |
|---|---|
| Accuracy | 0.7546 |
| Precision | 0.7579 |
| Recall | 0.7482 |
| F1 Score | 0.7531 |
| ROC-AUC | 0.8289 |

| Class | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| Negative | 0.75 | 0.76 | 0.76 | 12,500 |
| Positive | 0.76 | 0.75 | 0.75 | 12,500 |

## Observations
- **Balanced performance across both classes (~75%)** — no class imbalance issue here since IMDB is a 50/50 split, unlike the ANN's Wine Quality task.
- **Noticeably weaker than CNN's 92.5%** on its own task — this is expected and is a core point for comparison, not a flaw in implementation. Vanilla RNNs suffer from **vanishing gradients** over long sequences: with `MAX_LEN=200`, the model has to propagate signal from early words all the way to the final hidden state, and plain RNN cells lose most of that information by the later timesteps.
- **ROC-AUC of 0.83** shows the model has learned real signal (well above the 0.5 random baseline) but confirms there's a meaningful ceiling on how well a vanilla RNN can model long-range dependencies in text.
- **This motivates the next project (LSTM)** on the identical dataset, preprocessing, and hyperparameters — the only architectural change is swapping `nn.RNN` for `nn.LSTM`, isolating the effect of the gating mechanism on long-sequence performance.