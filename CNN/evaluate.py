import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "cnn_model.pth")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def main():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))
    ])
    test_set = datasets.FashionMNIST(DATA_DIR, train=False, download=True, transform=transform)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

    model = CNN(num_classes=10).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    metrics = {
        "accuracy": accuracy_score(all_labels, all_preds),
        "precision_macro": precision_score(all_labels, all_preds, average="macro"),
        "recall_macro": recall_score(all_labels, all_preds, average="macro"),
        "f1_macro": f1_score(all_labels, all_preds, average="macro"),
    }

    print("=== Evaluation Metrics ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    print("\n=== Classification Report ===")
    print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES))

    with open(os.path.join(RESULTS_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title("Confusion Matrix - CNN (Fashion-MNIST)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"))
    plt.close()

    print(f"\nSaved metrics.json and confusion_matrix.png to {RESULTS_DIR}")


if __name__ == "__main__":
    main()