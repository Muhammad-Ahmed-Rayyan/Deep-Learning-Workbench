import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "winequality-red.csv")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "ann_model.keras")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
RANDOM_STATE = 42
EPOCHS = 100
BATCH_SIZE = 32

os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


def load_data():
    """Download the dataset if not already present, then load it."""
    if not os.path.exists(DATA_PATH):
        print("Downloading Wine Quality dataset...")
        df = pd.read_csv(DATA_URL, sep=";")
        df.to_csv(DATA_PATH, index=False)
    else:
        df = pd.read_csv(DATA_PATH)
    return df


def preprocess(df):
    """Convert quality score into binary label and split/scale features."""
    df = df.copy()
    df["label"] = (df["quality"] >= 7).astype(int)  # 1 = good wine, 0 = not good

    X = df.drop(columns=["quality", "label"]).values
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler


def build_model(input_dim):
    """A simple fully-connected ANN with dropout for regularization."""
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(32, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(16, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def plot_history(history):
    """Save training/validation accuracy and loss curves."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history["accuracy"], label="Train Accuracy")
    axes[0].plot(history.history["val_accuracy"], label="Val Accuracy")
    axes[0].set_title("Accuracy over Epochs")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()

    axes[1].plot(history.history["loss"], label="Train Loss")
    axes[1].plot(history.history["val_loss"], label="Val Loss")
    axes[1].set_title("Loss over Epochs")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()

    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, "training_curves.png")
    plt.savefig(out_path)
    print(f"Saved training curves to {out_path}")


def main():
    df = load_data()
    print(f"Dataset shape: {df.shape}")

    X_train, X_test, y_train, y_test, scaler = preprocess(df)

    model = build_model(input_dim=X_train.shape[1])
    model.summary()

    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True
    )

    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stop],
        verbose=1,
    )

    plot_history(history)

    model.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

    # Save test set + scaler params for evaluate.py to reuse
    np.savez(
        os.path.join(RESULTS_DIR, "test_data.npz"),
        X_test=X_test, y_test=y_test
    )
    print("Saved test data for evaluation step.")


if __name__ == "__main__":
    main()