"""
Dataset and feature-cache support for MLP classifier training.

This module never calls TextToSQLGenerator. Training uses the prompt,
SQL query, and malicious label already present in the CSV dataset.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from config import EXPECTED_FEATURE_DIM
from pipeline.codebert import UnixCoderEncoder
from pipeline.encoder import FeatureEncoder
from pipeline.input import InputReader
from pipeline.llm_checker import LLMChecker


class SQLSecurityDataset(Dataset):
    """
    Materializes combined feature vectors and labels for MLP training.

    Expensive frozen feature generators (Qwen, UniXcoder, MiniLM) run
    once during dataset construction when no cache is available. Loaded
    samples are returned as tensors compatible with DataLoader.
    """

    def __init__(
        self,
        csv_path: str | Path,
        codebert_encoder: UnixCoderEncoder,
        llm_checker: LLMChecker,
        feature_encoder: FeatureEncoder,
        cache_path: Optional[str | Path] = None,
        use_cache: bool = True,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.cache_path = Path(cache_path) if cache_path else None
        self.codebert_encoder = codebert_encoder
        self.llm_checker = llm_checker
        self.feature_encoder = feature_encoder

        if use_cache and self.cache_path and self.cache_path.exists():
            self.features, self.labels = self._load_cache(self.cache_path)
        else:
            self.features, self.labels = self._build_features()
            if use_cache and self.cache_path:
                self._save_cache(self.cache_path, self.features, self.labels)

    def _load_cache(self, cache_path: Path) -> tuple[np.ndarray, np.ndarray]:
        data = np.load(cache_path)
        features = np.asarray(data["features"], dtype=np.float32)
        labels = np.asarray(data["labels"], dtype=np.float32)
        self._validate_arrays(features, labels)
        return features, labels

    def _save_cache(
        self, cache_path: Path, features: np.ndarray, labels: np.ndarray
    ) -> None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, features=features, labels=labels)

    def _build_features(self) -> tuple[np.ndarray, np.ndarray]:
        reader = InputReader(self.csv_path)
        feature_rows: list[np.ndarray] = []
        labels: list[float] = []

        for sample in reader.read_prompts_train():
            with torch.no_grad():
                llm_output = self.llm_checker.analyze(sample.prompt, sample.sql_query)
                codebert_output = self.codebert_encoder.encode(sample.sql_query)
                combined = self.feature_encoder.encode(codebert_output, llm_output)

            vector = np.asarray(combined.vector, dtype=np.float32).reshape(-1)
            if vector.shape != (EXPECTED_FEATURE_DIM,):
                raise ValueError(
                    f"Feature vector for sample has shape {vector.shape}; "
                    f"expected ({EXPECTED_FEATURE_DIM},)"
                )

            feature_rows.append(vector)
            labels.append(float(sample.malicious))

        if not feature_rows:
            raise ValueError(f"No valid training samples found in {self.csv_path}")

        features = np.stack(feature_rows).astype(np.float32)
        label_array = np.asarray(labels, dtype=np.float32)
        self._validate_arrays(features, label_array)
        return features, label_array

    def _validate_arrays(self, features: np.ndarray, labels: np.ndarray) -> None:
        if features.ndim != 2 or features.shape[1] != EXPECTED_FEATURE_DIM:
            raise ValueError(
                f"Cached/generated features must have shape (N, {EXPECTED_FEATURE_DIM}); "
                f"got {features.shape}"
            )
        if labels.ndim != 1 or labels.shape[0] != features.shape[0]:
            raise ValueError(
                f"Labels must have shape ({features.shape[0]},); got {labels.shape}"
            )

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        feature = torch.from_numpy(self.features[index]).float()
        label = torch.tensor(self.labels[index], dtype=torch.float32)
        return feature, label
