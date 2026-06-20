"""
final_classifier.py

Final stage of the SQL Security pipeline. Consumes a combined_feature
(1152-d vector produced by encoder.FeatureEncoder) and produces a
prediction (label + confidence) indicating whether the underlying SQL
query is malicious.

Pipeline:

    combined_feature
          |
          v
    MLPClassifier (PyTorch, outputs raw logit)
          |
          v
    sigmoid -> probability -> prediction

This module performs ONLY inference. It does not train, does not call
Qwen/UniXcoder/MiniLM, does not parse SQL, and does not modify feature
vectors beyond shape/type conversion required for the model forward
pass.
"""

from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from schemas import combined_feature, prediction
from config import EXPECTED_FEATURE_DIM, FINAL_CLASSIFIER_PATH, DEVICE


class MLPClassifier(nn.Module):
    """
    Small feed-forward network mapping a 1152-d combined_feature vector
    to a single malicious-probability score in [0, 1].

    Architecture:
        Linear(1152, 256) -> ReLU
        Linear(256, 64)   -> ReLU -> Dropout(0.2)
        Linear(64, 1)     (raw logit; apply torch.sigmoid() at inference)

    Outputs raw logits rather than probabilities so that training can
    use the more numerically stable nn.BCEWithLogitsLoss(). Callers
    needing a probability (e.g. FinalClassifier.predict_proba) must
    apply torch.sigmoid() to the output themselves.
    """

    def __init__(self, input_dim: int = EXPECTED_FEATURE_DIM) -> None:
        """
        Args:
            input_dim: Dimensionality of the input feature vector.
                Defaults to EXPECTED_FEATURE_DIM (1152) from config.py.
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
        Args:
            x: FloatTensor of shape (batch_size, input_dim).

        Returns:
            FloatTensor of shape (batch_size, 1) containing raw logits
            (not yet passed through sigmoid).
        """
        return self.net(x)


class FinalClassifier:
    """
    Inference-only wrapper around MLPClassifier. Loads trained weights
    once and exposes preprocess / predict_proba / predict for use by
    the rest of the pipeline.
    """

    def __init__(self) -> None:
        """
        Initialize the FinalClassifier: builds the MLPClassifier,
        loads its weights from FINAL_CLASSIFIER_PATH (config.py),
        moves it to DEVICE, and sets it to eval mode.
        """
        self.device = torch.device(DEVICE if DEVICE else "cpu")
        self.model = MLPClassifier(input_dim=EXPECTED_FEATURE_DIM)
        self.load_model()

    def load_model(self) -> None:
        """
        Load saved model weights from FINAL_CLASSIFIER_PATH, move the
        model to self.device, and set it to eval mode.

        Raises:
            RuntimeError: if the weights file cannot be loaded.
        """
        if not FINAL_CLASSIFIER_PATH:
            raise RuntimeError(
                "FINAL_CLASSIFIER_PATH is not set in config.py. "
                "Add FINAL_CLASSIFIER_PATH = '<path to .pt weights>' "
                "to config.py before instantiating FinalClassifier."
            )

        try:
            state_dict = torch.load(
                FINAL_CLASSIFIER_PATH, map_location=self.device
            )
            self.model.load_state_dict(state_dict)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load final classifier weights from "
                f"'{FINAL_CLASSIFIER_PATH}': {exc}"
            ) from exc

        self.model.to(self.device)
        self.model.eval()

    def preprocess(self, feature: combined_feature) -> torch.Tensor:
        """
        Validate and convert a combined_feature into a model-ready
        tensor.

        Args:
            feature: combined_feature with .vector of shape (1152,).

        Returns:
            torch.FloatTensor of shape (1, 1152), moved to self.device.

        Raises:
            ValueError: if feature.vector does not have shape (1152,).
        """
        vector = np.asarray(feature.vector, dtype=np.float32).reshape(-1)

        if vector.shape != (EXPECTED_FEATURE_DIM,):
            raise ValueError(
                f"combined_feature.vector has unexpected shape: "
                f"expected ({EXPECTED_FEATURE_DIM},), got {vector.shape}"
            )

        tensor = torch.from_numpy(vector).float().unsqueeze(0)
        return tensor.to(self.device)

    def predict_proba(self, feature: combined_feature) -> float:
        """
        Run a forward pass and return the malicious probability.

        The model outputs a raw logit; sigmoid is applied here (rather
        than inside the model) for numerical stability during training
        with nn.BCEWithLogitsLoss().

        Args:
            feature: combined_feature with .vector of shape (1152,).

        Returns:
            float in [0.0, 1.0]: probability that the SQL query is
            malicious.
        """
        tensor = self.preprocess(feature)

        with torch.no_grad():
            logit = self.model(tensor)
            probability = torch.sigmoid(logit)

        return float(probability.squeeze().item())

    def predict(self, feature: combined_feature) -> prediction:
        """
        Classify a combined_feature as malicious (1) or benign (0).

        Args:
            feature: combined_feature with .vector of shape (1152,).

        Returns:
            prediction(label=0 or 1, confidence=probability), where
            label=1 (malicious) if probability >= 0.5, else label=0
            (benign).
        """
        probability = self.predict_proba(feature)
        label = 1 if probability >= 0.5 else 0
        return prediction(label=label, confidence=probability)


if __name__ == "__main__":
    # Local smoke test: build a random combined_feature and run it
    # through the classifier. Requires a valid FINAL_CLASSIFIER_PATH
    # in config.py pointing to trained weights.
    dummy_vector = np.random.rand(EXPECTED_FEATURE_DIM).astype(np.float32)
    dummy_feature = combined_feature(vector=dummy_vector)

    classifier = FinalClassifier()
    result = classifier.predict(dummy_feature)

    print(f"label={result.label} confidence={result.confidence:.4f}")