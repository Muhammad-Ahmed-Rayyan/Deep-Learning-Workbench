import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, roc_auc_score
)
from datasets import load_dataset

# Reuse the same tokenizer/dataset/model classes as train_colab.py
import re
from collections import Counter
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

MAX_LEN = 200
EMBED_DIM = 128
HIDDEN_DIM = 128
PAD_IDX = 0
UNK_IDX = 1

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def simple_tokenize(text):
    text = text.lower()
    text = re.sub(r"<br\s*/?>", " ", text)
    text = re.sub(r"[^a-z0-9' ]", " ", text)
    return text.split()


def encode(text, vocab, max_len):
    tokens = simple_tokenize(text)
    ids = [vocab.get(tok, UNK_IDX) for tok in tokens[:max_len]]
    return ids


class IMDBDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        ids = encode(self.texts[idx], self.vocab, self.max_len)
        return torch.tensor(ids, dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.float)


def collate_batch(batch):
    sequences, labels = zip(*batch)
    lengths = torch.tensor([len(seq) for seq in sequences], dtype=torch.long)
    padded = pad_sequence(sequences, batch_first=True, padding_value=PAD_IDX)
    labels = torch.stack(labels)
    return padded, lengths, labels


class RNNClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_IDX)
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x, lengths):
        embedded = self.embedding(x)
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, hidden = self.rnn(packed)
        hidden = self.dropout(hidden.squeeze(0))
        return self.fc(hidden).squeeze(1)


def main():
    with open(os.path.join(MODEL_DIR, "vocab.json"), "r") as f:
        vocab = json.load(f)

    dataset = load_dataset("stanfordnlp/imdb", cache_dir=DATA_DIR)
    test_texts = dataset["test"]["text"]
    test_labels = dataset["test"]["label"]

    test_set = IMDBDataset(test_texts, test_labels, vocab, MAX_LEN)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False, collate_fn=collate_batch)

    model = RNNClassifier(len(vocab), EMBED_DIM, HIDDEN_DIM).to(DEVICE)
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "rnn_model.pth"), map_location=DEVICE))
    model.eval()

    all_preds, all_probs, all_labels = [], [], []
    with torch.no_grad():
        for sequences, lengths, labels in test_loader:
            sequences = sequences.to(DEVICE)
            outputs = model(sequences, lengths)
            probs = torch.sigmoid(outputs).cpu().numpy()
            preds = (probs >= 0.5).astype(int)

            all_preds.extend(preds)
            all_probs.extend(probs)
            all_labels.extend(labels.numpy())

    metrics = {
        "accuracy": accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds),
        "recall": recall_score(all_labels, all_preds),
        "f1_score": f1_score(all_labels, all_preds),
        "roc_auc": roc_auc_score(all_labels, all_probs),
    }

    print("=== Evaluation Metrics ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    print("\n=== Classification Report ===")
    print(classification_report(all_labels, all_preds, target_names=["Negative", "Positive"]))

    with open(os.path.join(RESULTS_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Negative", "Positive"],
                yticklabels=["Negative", "Positive"])
    plt.title("Confusion Matrix - RNN (IMDB)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"))
    plt.close()

    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {metrics['roc_auc']:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - RNN")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "roc_curve.png"))
    plt.close()

    print(f"\nSaved metrics.json, confusion_matrix.png, roc_curve.png to {RESULTS_DIR}")


if __name__ == "__main__":
    main()