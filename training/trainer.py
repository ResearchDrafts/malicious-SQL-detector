"""Training loop for the SQL security MLP classifier."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn


class MLPTrainer:
    """
    Train, validate, evaluate, and checkpoint an MLPClassifier.
    """

    def __init__(
        self,
        model: nn.Module,
        device: str | torch.device = "cpu",
        learning_rate: float = 1e-3,
        patience: int = 5,
    ) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.patience = patience
        self.history: dict[str, list[float]] = {
            "train_loss": [],
            "val_loss": [],
            "val_accuracy": [],
            "val_precision": [],
            "val_recall": [],
            "val_f1": [],
        }
        self.best_state_dict: dict[str, Any] | None = None

    def train(self, train_loader, val_loader, epochs: int = 50) -> dict[str, list[float]]:
        best_val_loss = float("inf")
        epochs_without_improvement = 0

        for _epoch in range(epochs):
            train_loss = self._train_one_epoch(train_loader)
            val_metrics = self.evaluate(val_loader)

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["val_accuracy"].append(val_metrics["accuracy"])
            self.history["val_precision"].append(val_metrics["precision"])
            self.history["val_recall"].append(val_metrics["recall"])
            self.history["val_f1"].append(val_metrics["f1"])

            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                epochs_without_improvement = 0
                self.best_state_dict = deepcopy(self.model.state_dict())
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= self.patience:
                    break

        if self.best_state_dict is not None:
            self.model.load_state_dict(self.best_state_dict)

        return self.history

    def _train_one_epoch(self, train_loader) -> float:
        self.model.train()
        total_loss = 0.0
        total_samples = 0

        for features, labels in train_loader:
            features = features.to(self.device)
            labels = labels.to(self.device).float()

            self.optimizer.zero_grad()
            logits = self.model(features).view(-1)
            loss = self.criterion(logits, labels)
            loss.backward()
            self.optimizer.step()

            batch_size = labels.shape[0]
            total_loss += float(loss.item()) * batch_size
            total_samples += batch_size

        return total_loss / max(total_samples, 1)

    def evaluate(self, data_loader) -> dict[str, Any]:
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        all_labels: list[np.ndarray] = []
        all_predictions: list[np.ndarray] = []

        with torch.no_grad():
            for features, labels in data_loader:
                features = features.to(self.device)
                labels = labels.to(self.device).float()

                logits = self.model(features).view(-1)
                loss = self.criterion(logits, labels)
                probabilities = torch.sigmoid(logits)
                predictions = (probabilities >= 0.5).long()

                batch_size = labels.shape[0]
                total_loss += float(loss.item()) * batch_size
                total_samples += batch_size
                all_labels.append(labels.cpu().numpy().astype(np.int64))
                all_predictions.append(predictions.cpu().numpy().astype(np.int64))

        y_true = np.concatenate(all_labels) if all_labels else np.array([], dtype=np.int64)
        y_pred = (
            np.concatenate(all_predictions)
            if all_predictions
            else np.array([], dtype=np.int64)
        )

        tn = int(np.sum((y_true == 0) & (y_pred == 0)))
        fp = int(np.sum((y_true == 0) & (y_pred == 1)))
        fn = int(np.sum((y_true == 1) & (y_pred == 0)))
        tp = int(np.sum((y_true == 1) & (y_pred == 1)))

        accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)

        return {
            "loss": total_loss / max(total_samples, 1),
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "confusion_matrix": np.array([[tn, fp], [fn, tp]], dtype=np.int64),
        }

    def save_checkpoint(self, path: str | Path) -> None:
        checkpoint_path = Path(path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), checkpoint_path)

    def plot_training_history(self, save_path: str | Path) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(self.history["train_loss"], label="train loss")
        ax.plot(self.history["val_loss"], label="validation loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(save_path)
        plt.close(fig)
