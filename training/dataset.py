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
        checkpoint_interval: int = 100,
        progress_interval: int = 25,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.cache_path = Path(cache_path) if cache_path else None
        self.partial_cache_path = (
            self.cache_path.with_name(f"{self.cache_path.stem}.partial.npz")
            if self.cache_path
            else None
        )
        self.use_cache = use_cache
        self.checkpoint_interval = max(1, int(checkpoint_interval))
        self.progress_interval = max(1, int(progress_interval))
        self.codebert_encoder = codebert_encoder
        self.llm_checker = llm_checker
        self.feature_encoder = feature_encoder

        if use_cache and self.cache_path and self.cache_path.exists():
            self.features, self.labels = self._load_cache(self.cache_path)
        else:
            self.features, self.labels = self._build_features()
            if use_cache and self.cache_path:
                self._save_cache(self.cache_path, self.features, self.labels)
                if self.partial_cache_path and self.partial_cache_path.exists():
                    self.partial_cache_path.unlink()

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

    def _load_partial_cache(self) -> tuple[list[np.ndarray], list[float], int]:
        if not self.partial_cache_path or not self.partial_cache_path.exists():
            return [], [], 0

        data = np.load(self.partial_cache_path)
        features = np.asarray(data["features"], dtype=np.float32)
        labels = np.asarray(data["labels"], dtype=np.float32)
        processed_count = int(data["processed_count"])

        if processed_count != features.shape[0]:
            raise ValueError(
                "Partial feature cache is inconsistent: processed_count "
                f"is {processed_count}, but it contains {features.shape[0]} rows"
            )
        if features.shape != (processed_count, EXPECTED_FEATURE_DIM):
            raise ValueError(
                "Partial feature cache has unexpected shape: "
                f"{features.shape}"
            )
        if labels.shape != (processed_count,):
            raise ValueError(
                f"Partial label cache has unexpected shape: {labels.shape}"
            )

        return list(features), labels.tolist(), processed_count

    def _save_partial_cache(
        self,
        feature_rows: list[np.ndarray],
        labels: list[float],
        processed_count: int,
    ) -> None:
        if not self.use_cache or not self.partial_cache_path:
            return

        self.partial_cache_path.parent.mkdir(parents=True, exist_ok=True)
        features = (
            np.stack(feature_rows).astype(np.float32)
            if feature_rows
            else np.empty((0, EXPECTED_FEATURE_DIM), dtype=np.float32)
        )
        label_array = np.asarray(labels, dtype=np.float32)
        temporary_path = self.partial_cache_path.with_name(
            f".{self.partial_cache_path.stem}.tmp.npz"
        )
        np.savez_compressed(
            temporary_path,
            features=features,
            labels=label_array,
            processed_count=np.asarray(processed_count, dtype=np.int64),
        )
        temporary_path.replace(self.partial_cache_path)

    def _build_features(self) -> tuple[np.ndarray, np.ndarray]:
        reader = InputReader(self.csv_path)
        samples = list(reader.read_prompts_train())
        feature_rows, labels, start_index = self._load_partial_cache()

        if start_index > len(samples):
            raise ValueError(
                f"Partial cache contains {start_index} rows, but the dataset "
                f"contains only {len(samples)} samples"
            )
        if start_index:
            print(
                f"Resuming feature generation for {self.csv_path} at "
                f"sample {start_index + 1}/{len(samples)}"
            )

        for index in range(start_index, len(samples)):
            sample = samples[index]
            try:
                with torch.no_grad():
                    llm_output = self.llm_checker.analyze(
                        sample.prompt, sample.sql_query
                    )
                    codebert_output = self.codebert_encoder.encode(sample.sql_query)
                    combined = self.feature_encoder.encode(
                        codebert_output, llm_output
                    )
            except Exception as exc:
                self._save_partial_cache(feature_rows, labels, index)
                raise RuntimeError(
                    f"Feature generation failed at sample {index + 1}/"
                    f"{len(samples)} in {self.csv_path}: {exc}"
                ) from exc

            vector = np.asarray(combined.vector, dtype=np.float32).reshape(-1)
            if vector.shape != (EXPECTED_FEATURE_DIM,):
                raise ValueError(
                    f"Feature vector for sample has shape {vector.shape}; "
                    f"expected ({EXPECTED_FEATURE_DIM},)"
                )

            feature_rows.append(vector)
            labels.append(float(sample.malicious))

            processed_count = index + 1
            if processed_count % self.checkpoint_interval == 0:
                self._save_partial_cache(
                    feature_rows, labels, processed_count
                )
                print(
                    f"Cached {processed_count}/{len(samples)} features "
                    f"for {self.csv_path}"
                )
            elif processed_count % self.progress_interval == 0:
                print(
                    f"Generated {processed_count}/{len(samples)} features "
                    f"for {self.csv_path}"
                )

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
