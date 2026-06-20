"""
encoder.py

FeatureEncoder: combines codebert_output (UniXcoder embedding) and
LLM_output (Qwen raw reasoning) into a single 1152-dimensional
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

Feature layout (1152-d total):

    768  UniXcoder embedding              (codebert_output.embedding)
    384  MiniLM embedding of Qwen's raw reasoning text (build_reason_text -> encode_reason)

This is a pure embedding-fusion design: no scalar decision features
(UniXcoder probability, Qwen malicious/ambiguous flags, Qwen
confidence) are included, since those are model predictions rather
than semantic representations. The final classifier learns directly
from the embeddings themselves.

This module performs ONLY feature encoding. It does not call Qwen, does
not load UniXcoder, does not parse SQL, and does not perform final
classification.
"""

from __future__ import annotations
import numpy as np
from sentence_transformers import SentenceTransformer
from schemas import LLM_output, codebert_output, llm_feature, combined_feature
from config import DEVICE, UNIXCODER_EMBEDDING_DIM, MINILM_EMBEDDING_DIM, EXPECTED_FEATURE_DIM, MINILM_MODEL_NAME

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

    def build_llm_feature(self, reason_embedding: np.ndarray) -> llm_feature:
        """
        Wrap a MiniLM reasoning embedding into an llm_feature object.

        Args:
            reason_embedding: MiniLM embedding from encode_reason(),
                shape (384,).

        Returns:
            llm_feature wrapping the (384,) np.ndarray embedding.
        """
        return llm_feature(vector=reason_embedding)
    
    def build_reason_text(self, llm_output: LLM_output) -> str:
        """
        Extract the reasoning text to embed with MiniLM.

        Uses llm_output.raw_response (Qwen's original, unparsed
        response text) directly as the semantic reasoning signal,
        rather than reconstructing a synthetic summary from
        malicious_reason/ambiguity_reason. This preserves the full
        nuance of Qwen's reasoning instead of collapsing it into two
        short fields.

        Args:
            llm_output: LLM_output produced by LLMChecker, containing
                raw_response (Qwen's original output text).

        Returns:
            The stripped raw_response string, or "benign request" if
            raw_response is empty, whitespace-only, or missing.
        """
        raw_response = (llm_output.raw_response or "").strip()

        if not raw_response:
            return "benign request"

        else:
            raise ValueError("Qwen returned empty reasoning response")

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
        llm_feature_obj: llm_feature,
    ) -> np.ndarray:
        """
        Concatenate embedding components into a single 1152-d vector.

        Layout:
            [0:768]    codebert_output.embedding (UniXcoder CLS embedding)
            [768:1152] llm_feature_obj.vector (MiniLM embedding of Qwen's raw reasoning text)

        This is a pure embedding-fusion vector. No scalar decision
        features (probability, malicious flag, ambiguous flag,
        confidence) are included.

        Args:
            codebert_output: Output from UnixCoderEncoder.encode().
            llm_feature_obj: llm_feature produced by build_llm_feature(),
                wrapping a (384,) MiniLM reasoning embedding.

        Returns:
            np.ndarray of shape (1152,), dtype float32.

        Raises:
            ValueError: if the resulting vector does not have shape
                (1152,).
        """
        unixcoder_embedding = np.asarray(
            codebert_output.embedding, dtype=np.float32
        ).reshape(-1)
        reason_embedding = np.asarray(
            llm_feature_obj.vector, dtype=np.float32
        ).reshape(-1)

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

        feature_vector = np.concatenate(
            [unixcoder_embedding, reason_embedding]
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

            build_reason_text -> encode_reason -> build_llm_feature
            -> build_feature_vector

        Args:
            codebert_output: Output from UnixCoderEncoder.encode().
            llm_output: Output from LLMChecker.analyze().

        Returns:
            combined_feature wrapping a (1152,) np.ndarray, ready to be
            passed to the final classifier.
        """
        reason_text = self.build_reason_text(llm_output)
        reason_embedding = self.encode_reason(reason_text)
        llm_feature_obj = self.build_llm_feature(reason_embedding)
        feature_vector = self.build_feature_vector(
            codebert_output, llm_feature_obj
        )

        return combined_feature(vector=feature_vector)