import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, roc_auc_score
)
from tensorflow import keras

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "ann_model.keras")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
TEST_DATA_PATH = os.path.join(RESULTS_DIR, "test_data.npz")


def main():
    # Load model and test data
    model = keras.models.load_model(MODEL_PATH)
    data = np.load(TEST_DATA_PATH)
    X_test, y_test = data["X_test"], data["y_test"]

    # Predictions
    y_probs = model.predict(X_test).flatten()
    y_pred = (y_probs >= 0.5).astype(int)

    # Metrics
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_probs),
    }

    print("=== Evaluation Metrics ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred))

    # Save metrics as JSON
    with open(os.path.join(RESULTS_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    # Confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Not Good", "Good"],
                yticklabels=["Not Good", "Good"])
    plt.title("Confusion Matrix - ANN")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"))
    plt.close()

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_probs)
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {metrics['roc_auc']:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - ANN")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "roc_curve.png"))
    plt.close()

    print(f"\nSaved metrics.json, confusion_matrix.png, roc_curve.png to {RESULTS_DIR}")


if __name__ == "__main__":
    main()