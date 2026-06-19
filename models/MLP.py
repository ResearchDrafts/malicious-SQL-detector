"""
Training utilities and model management for the MLP classifier.
Handles data loading, training loop with early stopping, evaluation metrics,
and checkpoint management.
"""

from __future__ import annotations
import csv
from pathlib import Path
from typing import Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt

from pipeline.codebert import UnixCoderEncoder
from pipeline.llm_checker import LLMChecker
from pipeline.encoder import FeatureEncoder
from schemas import input_prompt, sql_result, codebert_output, combined_feature
from config import DEVICE, EXPECTED_FEATURE_DIM, FINAL_CLASSIFIER_PATH


class SQLSecurityDataset(Dataset):
    """
    PyTorch Dataset for SQL security classification.
    Reads prompt and SQL query pairs, generates embeddings via the pipeline.
    """

    def __init__(
        self,
        csv_path: str | Path,
        codebert_encoder: UnixCoderEncoder,
        llm_checker: LLMChecker,
        feature_encoder: FeatureEncoder,
    ) -> None:
        """
        Initialize the dataset.

        Args:
            csv_path: Path to CSV file with columns:
                - "Malicious or not" (label: 0/1)
                - "prompt_entered" (user prompt)
                - "SQL query" (generated SQL)
            codebert_encoder: UnixCoderEncoder instance for SQL embedding.
            llm_checker: LLMChecker instance for prompt analysis.
            feature_encoder: FeatureEncoder instance for feature vector construction.
        """
        self.csv_path = Path(csv_path)
        self.codebert_encoder = codebert_encoder
        self.llm_checker = llm_checker
        self.feature_encoder = feature_encoder

        self.data = self._load_csv()

    def _load_csv(self) -> list[dict]:
        """
        Load CSV file and parse rows.

        Returns:
            List of dicts with keys: label, prompt, sql_query.
        """
        rows = []
        try:
            with open(self.csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    label = int(row.get("Malicious or not", 0))
                    prompt = row.get("prompt_entered", "").strip()
                    sql_query = row.get("SQL query", "").strip()

                    if prompt and sql_query:
                        rows.append({
                            "label": label,
                            "prompt": prompt,
                            "sql_query": sql_query,
                        })
        except Exception as exc:
            raise ValueError(f"Failed to load CSV {self.csv_path}: {exc}") from exc

        return rows

    def __len__(self) -> int:
        """Return total number of samples."""
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a single sample and return embedded feature vector + label.

        Args:
            idx: Sample index.

        Returns:
            Tuple (feature_tensor, label) where feature_tensor is (1156,)
            and label is 0 or 1.
        """
        row = self.data[idx]
        label = row["label"]
        prompt = row["prompt"]
        sql_query = row["sql_query"]

        try:
            codebert_out = self.codebert_encoder.encode(sql_query)
            llm_out = self.llm_checker.analyze(prompt, sql_query)
            combined_feat = self.feature_encoder.encode(codebert_out, llm_out)

            feature_vector = np.asarray(combined_feat.vector, dtype=np.float32)
            feature_tensor = torch.from_numpy(feature_vector).float()

            return feature_tensor, label

        except Exception as exc:
            raise RuntimeError(
                f"Failed to process sample {idx}: prompt='{prompt}': {exc}"
            ) from exc


class MLPClassifier(nn.Module):
    """
    Feed-forward neural network for binary SQL security classification.

    Architecture:
        Linear(1156, 256) -> ReLU
        Linear(256, 64)   -> ReLU -> Dropout(0.2)
        Linear(64, 1)     -> outputs raw logit
    """

    def __init__(self, input_dim: int = EXPECTED_FEATURE_DIM) -> None:
        """
        Args:
            input_dim: Dimensionality of input features (1156).
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, 1156).

        Returns:
            Raw logits of shape (batch_size, 1).
        """
        return self.net(x)


class MLPTrainer:
    """
    Trainer class for the MLP classifier.
    Handles training loops, validation, early stopping, and metrics computation.
    """

    def __init__(
        self,
        model: MLPClassifier,
        device: str = "cpu",
        learning_rate: float = 0.001,
        patience: int = 5,
    ) -> None:
        """
        Initialize the trainer.

        Args:
            model: MLPClassifier instance.
            device: Device to train on ("cpu" or "cuda").
            learning_rate: Learning rate for Adam optimizer.
            patience: Number of epochs with no improvement before early stopping.
        """
        self.model = model.to(device)
        self.device = torch.device(device)
        self.optimizer = Adam(model.parameters(), lr=learning_rate)
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.patience = patience
        self.best_val_loss = float("inf")
        self.patience_counter = 0

        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []

    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """
        Train for one epoch.

        Args:
            train_loader: DataLoader for training data.

        Returns:
            Tuple (average_loss, accuracy).
        """
        self.model.train()
        total_loss = 0.0
        all_preds = []
        all_labels = []

        for features, labels in train_loader:
            features = features.to(self.device)
            labels = labels.to(self.device).float().unsqueeze(1)

            self.optimizer.zero_grad()
            logits = self.model(features)
            loss = self.loss_fn(logits, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

            probs = torch.sigmoid(logits).detach().cpu().numpy()
            preds = (probs >= 0.5).astype(int).flatten()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy().flatten().astype(int))

        avg_loss = total_loss / len(train_loader)
        accuracy = accuracy_score(all_labels, all_preds)

        return avg_loss, accuracy

    def validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """
        Validate on validation set.

        Args:
            val_loader: DataLoader for validation data.

        Returns:
            Tuple (average_loss, accuracy).
        """
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for features, labels in val_loader:
                features = features.to(self.device)
                labels = labels.to(self.device).float().unsqueeze(1)

                logits = self.model(features)
                loss = self.loss_fn(logits, labels)
                total_loss += loss.item()

                probs = torch.sigmoid(logits).cpu().numpy()
                preds = (probs >= 0.5).astype(int).flatten()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy().flatten().astype(int))

        avg_loss = total_loss / len(val_loader)
        accuracy = accuracy_score(all_labels, all_preds)

        return avg_loss, accuracy

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 50,
    ) -> dict:
        """
        Main training loop with early stopping.

        Args:
            train_loader: DataLoader for training data.
            val_loader: DataLoader for validation data.
            epochs: Maximum number of epochs.

        Returns:
            Dictionary with training history.
        """
        print(f"Starting training for up to {epochs} epochs...")
        print(f"Device: {self.device}, Patience: {self.patience}")

        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc = self.validate(val_loader)

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)

            print(
                f"Epoch {epoch + 1}/{epochs} | "
                f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}"
            )

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                print(f"  -> Validation loss improved. Saving checkpoint.")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    print(f"Early stopping triggered (patience={self.patience})")
                    break

        return {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "train_accuracies": self.train_accuracies,
            "val_accuracies": self.val_accuracies,
        }

    def evaluate(self, test_loader: DataLoader) -> dict:
        """
        Evaluate model on test set with full metrics.

        Args:
            test_loader: DataLoader for test data.

        Returns:
            Dictionary with metrics: accuracy, precision, recall, f1,
            confusion_matrix.
        """
        self.model.eval()
        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for features, labels in test_loader:
                features = features.to(self.device)
                logits = self.model(features)
                probs = torch.sigmoid(logits).cpu().numpy()
                preds = (probs >= 0.5).astype(int).flatten()

                all_probs.extend(probs.flatten())
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy().astype(int))

        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, zero_division=0)
        recall = recall_score(all_labels, all_preds, zero_division=0)
        f1 = f1_score(all_labels, all_preds, zero_division=0)
        cm = confusion_matrix(all_labels, all_preds)

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "confusion_matrix": cm,
            "predictions": all_preds,
            "probabilities": all_probs,
            "labels": all_labels,
        }

    def save_checkpoint(self, path: str | Path) -> None:
        """
        Save model weights to disk.

        Args:
            path: Path to save the model checkpoint.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        try:
            torch.save(self.model.state_dict(), path)
            print(f"Model checkpoint saved to {path}")
        except Exception as exc:
            raise RuntimeError(f"Failed to save model: {exc}") from exc

    def load_checkpoint(self, path: str | Path) -> None:
        """
        Load model weights from disk.

        Args:
            path: Path to the saved model checkpoint.
        """
        try:
            state_dict = torch.load(path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            print(f"Model checkpoint loaded from {path}")
        except Exception as exc:
            raise RuntimeError(f"Failed to load model: {exc}") from exc

    def plot_training_history(self, save_path: str | Path | None = None) -> None:
        """
        Plot training and validation loss/accuracy curves.

        Args:
            save_path: Optional path to save the figure.
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        axes[0].plot(self.train_losses, label="Train Loss")
        axes[0].plot(self.val_losses, label="Val Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("Training and Validation Loss")
        axes[0].legend()
        axes[0].grid(True)

        axes[1].plot(self.train_accuracies, label="Train Accuracy")
        axes[1].plot(self.val_accuracies, label="Val Accuracy")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy")
        axes[1].set_title("Training and Validation Accuracy")
        axes[1].legend()
        axes[1].grid(True)

        plt.tight_layout()

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150)
            print(f"Training history plot saved to {save_path}")
        else:
            plt.show()


def plot_confusion_matrix(cm: np.ndarray, save_path: str | Path | None = None) -> None:
    """
    Plot confusion matrix.

    Args:
        cm: Confusion matrix from sklearn.metrics.confusion_matrix.
        save_path: Optional path to save the figure.
    """
    fig, ax = plt.subplots(figsize=(6, 5))

    im = ax.imshow(cm, cmap="Blues", aspect="auto")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Benign", "Malicious"])
    ax.set_yticklabels(["Benign", "Malicious"])

    for i in range(2):
        for j in range(2):
            text = ax.text(j, i, cm[i, j], ha="center", va="center", color="white")

    plt.colorbar(im, ax=ax)
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"Confusion matrix plot saved to {save_path}")
    else:
        plt.show()