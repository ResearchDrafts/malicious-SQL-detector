"""
Extracts output from unixcoder (codeBERT model)
"""

from __future__ import annotations
import re
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn.functional as F
from schemas import codebert_output
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from config import DEVICE, MAX_SQL_LENGTH, UNIXCODER_PATH

class UnixCoderEncoder:
    """
    Wraps a fine-tuned UniXcoder sequence-classification model to produce,
    for a given SQL query string:
        1. A 768-dimensional embedding (CLS token of the final hidden layer).
        2. A scalar malicious-probability score from the classification head.
        
        does not perform any SQL-specific parsing beyond basic text cleanup.
    """

    def __init__(self) -> None:
        # Load the tokenizer and fine-tuned UniXcoder model from, UNIXCODER_PATH, move the model to DEVICE, and set it to eval mode.
        self.device = torch.device(DEVICE)
        self.tokenizer = AutoTokenizer.from_pretrained(UNIXCODER_PATH)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            UNIXCODER_PATH,
            output_hidden_states=True,
        )
        self.model.to(self.device)
        self.model.eval()

    def preprocess(self, sql_query: str) -> str:
        """
        Clean raw SQL text prior to tokenization.
        Strips leading/trailing whitespace and collapses any run of
        whitespace characters (spaces, tabs, newlines) into a single
        space. Handles None / empty input gracefully by returning an
        empty string rather than raising.
        
        Args: sql_query: Raw SQL query text.
        Returns: Cleaned SQL query text.
        """
        if not sql_query:
            return ""
        cleaned = sql_query.strip()
        cleaned = re.sub(r"\s+", " ", cleaned)
        return cleaned

    def tokenize(self, sql_query: str) -> dict[str, torch.Tensor]:
        """
        Tokenize a cleaned SQL query using the UniXcoder tokenizer.

        Args: sql_query: Cleaned SQL query text.
        Returns: Dictionary of tensors (input_ids, attention_mask, ...) moved to DEVICE, suitable for direct use as model(**inputs).
        """
        encoded = self.tokenizer(
            sql_query,
            truncation=True,
            padding=True,
            max_length=MAX_SQL_LENGTH,
            return_tensors="pt",
        )
        encoded = {key: tensor.to(self.device) for key, tensor in encoded.items()}
        return encoded

    def forward(self, inputs: dict[str, torch.Tensor]):
        """
        Run a forward pass through the model under torch.no_grad().

        Args: inputs: Tokenized inputs, as produced by tokenize().

        Returns: Raw HuggingFace model output object (contains .logits and .hidden_states since output_hidden_states=True).
        """
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs

    def get_embedding(self, outputs) -> np.ndarray:
        """
        Extract the CLS-token embedding from the final hidden layer.

        Args:
            outputs: Raw model output from forward(), with
                outputs.hidden_states populated.
        Returns:
            numpy array of shape (768,) representing the pooled
            sequence embedding.
        """
        last_hidden_state = outputs.hidden_states[-1]
        cls_embedding = last_hidden_state[:, 0, :]
        return cls_embedding.squeeze(0).cpu().numpy()

    def get_probability(self, outputs) -> float:
        """
        Extract the malicious-class probability from the model's logits.
        Assumes a binary classification head where index 1 corresponds
        to the "malicious" class. Applies softmax over the logits to
        convert them into a probability distribution.

        Args:
            outputs: Raw model output from forward(), with
                outputs.logits populated.

        Returns:
            Float in [0.0, 1.0] representing P(malicious).
        """
        logits = outputs.logits
        probs = F.softmax(logits, dim=-1)
        malicious_prob = probs[0, 1].item()
        return float(malicious_prob)

    def encode(self, sql_query: str) -> codebert_output:
        """
        Run the full pipeline on a single SQL query:
            preprocess -> tokenize -> forward -> get_embedding -> get_probability

        Args:
            sql_query: Raw SQL query string to analyze.

        Returns:
            codebert_output containing the 768-d embedding and the
            malicious probability.

        Notes:
            Empty or whitespace-only input is handled gracefully: it is
            still tokenized and passed through the model (an empty
            string tokenizes to just special tokens), so this method
            does not raise on degenerate input. Callers that want to
            special-case empty SQL should check sql_query themselves
            before calling encode().
        """
        cleaned = self.preprocess(sql_query)
        inputs = self.tokenize(cleaned)
        outputs = self.forward(inputs)

        embedding = self.get_embedding(outputs)
        probability = self.get_probability(outputs)

        return codebert_output(embedding=embedding, probability=probability)