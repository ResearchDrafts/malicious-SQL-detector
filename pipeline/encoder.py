"""
encoder.py

FeatureEncoder: combines LLM_output (Qwen reasoning) and codebert_output
(UniXcoder embedding + probability) into a single 1156-dimensional
combined_feature vector for the final classifier.

Pipeline:

    LLM_output
         +
    codebert_output
         |
         v
    FeatureEncoder
         |
         v
    combined_feature

Feature layout (1156-d total):

    768  UniXcoder embedding              (codebert_output.embedding)
    384  MiniLM embedding of LLM reasoning (build_reason_text -> encode_reason)
      1  UniXcoder malicious probability  (codebert_output.probability)
      1  Qwen malicious flag              (int(llm_output.malicious))
      1  Qwen ambiguous flag              (int(llm_output.ambiguous))
      1  Qwen confidence (normalized)     (llm_output.confidence / 100.0)

This module performs ONLY feature encoding. It does not call Qwen, does
not load UniXcoder, does not parse SQL, and does not perform final
classification.
"""

from __future__ import annotations
import numpy as np
from sentence_transformers import SentenceTransformer
from schemas import LLM_output, codebert_output, combined_feature
from config import DEVICE, UNIXCODER_EMBEDDING_DIM, MINILM_EMBEDDING_DIM, SCALAR_FEATURE_COUNT, EXPECTED_FEATURE_DIM, MINILM_MODEL_NAME 

class FeatureEncoder:
    """
    Combines Qwen-derived LLM_output and UniXcoder-derived codebert_output
    into a single fixed-length feature vector (combined_feature) for
    consumption by the final classifier.

    The MiniLM sentence-transformer model is loaded once at construction
    time and reused across all encode() calls.
    """

    def __init__(self) -> None:
        """
        Initialize the FeatureEncoder.

        Loads the MiniLM sentence-transformer model
        (sentence-transformers/all-MiniLM-L6-v2) once for reuse across
        all subsequent encode() calls.
        """
        if DEVICE:
            self.minilm = SentenceTransformer(MINILM_MODEL_NAME, device=DEVICE)
        else:
            self.minilm = SentenceTransformer(MINILM_MODEL_NAME)

    def build_reason_text(self, llm_output: LLM_output) -> str:
        """
        Construct a single reasoning text string from the LLM_output's
        malicious_reason and ambiguity_reason fields.

        Format (one line per present reason):
            "Malicious: <malicious_reason>"
            "Ambiguous: <ambiguity_reason>"

        If both reasons are empty/falsy, returns "benign request".

        Args:
            llm_output: LLM_output produced by LLMChecker.

        Returns:
            A single string summarizing the LLM's reasoning.
        """
        lines: list[str] = []

        malicious_reason = (llm_output.malicious_reason or "").strip()
        ambiguity_reason = (llm_output.ambiguity_reason or "").strip()

        if malicious_reason:
            lines.append(f"Malicious: {malicious_reason}")
        if ambiguity_reason:
            lines.append(f"Ambiguous: {ambiguity_reason}")

        if not lines:
            return "benign request"

        return "\n".join(lines)

    def encode_reason(self, reason_text: str) -> np.ndarray:
        """
        Encode a reason text string into a MiniLM sentence embedding.

        Args:
            reason_text: Text produced by build_reason_text().

        Returns:
            np.ndarray of shape (384,), dtype float32.
        """
        embedding = self.minilm.encode(
            reason_text,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        embedding = np.asarray(embedding, dtype=np.float32).reshape(-1)

        if embedding.shape[0] != MINILM_EMBEDDING_DIM:
            raise ValueError(
                f"MiniLM embedding has unexpected dimensionality: "
                f"expected {MINILM_EMBEDDING_DIM}, got {embedding.shape[0]}"
            )

        return embedding

    def build_feature_vector(
        self,
        codebert_output: codebert_output,
        llm_output: LLM_output,
        reason_embedding: np.ndarray,
    ) -> np.ndarray:
        """
        Concatenate all feature components into a single 1156-d vector.

        Layout:
            [0:768]    codebert_output.embedding (UniXcoder CLS embedding)
            [768:1152] reason_embedding (MiniLM embedding of LLM reasoning)
            [1152]     codebert_output.probability (UniXcoder malicious prob)
            [1153]     int(llm_output.malicious)   (Qwen malicious flag)
            [1154]     int(llm_output.ambiguous)   (Qwen ambiguous flag)
            [1155]     llm_output.confidence clipped to [0,100] then /100.0

        Args:
            codebert_output: Output from UnixCoderEncoder.encode().
            llm_output: Output from LLMChecker.analyze().
            reason_embedding: MiniLM embedding from encode_reason(),
                shape (384,).

        Returns:
            np.ndarray of shape (1156,), dtype float32.

        Raises:
            ValueError: if the resulting vector does not have shape
                (1156,).
        """
        unixcoder_embedding = np.asarray(
            codebert_output.embedding, dtype=np.float32
        ).reshape(-1)
        reason_embedding = np.asarray(reason_embedding, dtype=np.float32).reshape(-1)

        if unixcoder_embedding.shape[0] != UNIXCODER_EMBEDDING_DIM:
            raise ValueError(
                f"UniXcoder embedding has unexpected dimensionality: "
                f"expected {UNIXCODER_EMBEDDING_DIM}, got "
                f"{unixcoder_embedding.shape[0]}"
            )
        if reason_embedding.shape[0] != MINILM_EMBEDDING_DIM:
            raise ValueError(
                f"Reason embedding has unexpected dimensionality: "
                f"expected {MINILM_EMBEDDING_DIM}, got "
                f"{reason_embedding.shape[0]}"
            )

        probability = float(codebert_output.probability)
        if not 0.0 <= probability <= 1.0:
            raise ValueError(
                f"codebert_output.probability out of expected range [0,1]: "
                f"{probability}"
            )

        confidence = max(0.0, min(100.0, float(llm_output.confidence))) / 100.0

        scalars = np.array(
            [
                probability,
                float(int(llm_output.malicious)),
                float(int(llm_output.ambiguous)),
                confidence,
            ],
            dtype=np.float32,
        )

        feature_vector = np.concatenate(
            [unixcoder_embedding, reason_embedding, scalars]
        )

        if feature_vector.shape[0] != EXPECTED_FEATURE_DIM:
            raise ValueError(
                f"combined_feature vector has unexpected dimensionality: "
                f"expected {EXPECTED_FEATURE_DIM}, got "
                f"{feature_vector.shape[0]}"
            )

        return feature_vector

    def encode(
        self,
        codebert_output: codebert_output,
        llm_output: LLM_output,
    ) -> combined_feature:
        """
        Run the full feature-encoding pipeline:

            build_reason_text -> encode_reason -> build_feature_vector

        Args:
            codebert_output: Output from UnixCoderEncoder.encode().
            llm_output: Output from LLMChecker.analyze().

        Returns:
            combined_feature wrapping a (1156,) np.ndarray, ready to be
            passed to the final classifier.
        """
        reason_text = self.build_reason_text(llm_output)
        reason_embedding = self.encode_reason(reason_text)
        feature_vector = self.build_feature_vector(
            codebert_output, llm_output, reason_embedding
        )

        return combined_feature(vector=feature_vector)