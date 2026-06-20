"""
Text-to-SQL generation using fine-tuned LLaMA 3.2 adapter with PEFT.
Loads locally saved adapter weights and generates SQL from natural language prompts.
"""

from __future__ import annotations
import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer, GenerationConfig
from schemas import input_prompt, sql_result
from config import DEVICE, LLAMA_BASE_MODEL, LLAMA_ADAPTER_PATH, TEXT_TO_SQL_MAX_LENGTH, TEXT_TO_SQL_GENERATION_KWARGS


class TextToSQLGenerator:
    """
    Generates SQL queries from natural language prompts using a fine-tuned
    LLaMA 3.2 model with PEFT (Parameter-Efficient Fine-Tuning) adapter.

    The adapter weights are loaded from a local directory (LLAMA_ADAPTER_PATH).
    """

    def __init__(self) -> None:
        """
        Initialize the TextToSQLGenerator by loading the base model,
        PEFT adapter, and tokenizer.

        Raises:
            RuntimeError: if the model or adapter cannot be loaded.
        """
        self.device = torch.device(DEVICE if DEVICE else "cpu")
        self._load_model()
        self._load_tokenizer()

    def _load_model(self) -> None:
        """
        Load the base LLaMA model with the PEFT adapter from local storage.
        """
        try:
            self.model = AutoPeftModelForCausalLM.from_pretrained(
                LLAMA_ADAPTER_PATH,
                device_map="auto",
                torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
            )
            self.model = self.model.merge_and_unload()
            self.model.to(self.device)
            self.model.eval()
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load LLaMA adapter from '{LLAMA_ADAPTER_PATH}': {exc}"
            ) from exc

    def _load_tokenizer(self) -> None:
        """
        Load the tokenizer for the base LLaMA model.
        """
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(LLAMA_BASE_MODEL)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load tokenizer for '{LLAMA_BASE_MODEL}': {exc}"
            ) from exc

    def preprocess_prompt(self, prompt: str) -> str:
        """
        Format the user prompt into a structured instruction for SQL generation.

        Args:
            prompt: Raw natural language query.

        Returns:
            Formatted instruction string.
        """
        if not prompt:
            return ""

        system_instruction = (
            "You are a SQL expert. Generate a SQL query based on the user's request. "
            "Return ONLY the SQL query, no explanation.\n\n"
            "User request: "
        )
        return system_instruction + prompt.strip()

    def tokenize(self, text: str) -> dict[str, torch.Tensor]:
        """
        Tokenize input text using the LLaMA tokenizer.

        Args:
            text: Preprocessed instruction text.

        Returns:
            Dictionary of tokenized tensors moved to device.
        """
        encoded = self.tokenizer(
            text,
            truncation=True,
            padding=False,
            max_length=TEXT_TO_SQL_MAX_LENGTH,
            return_tensors="pt",
        )
        encoded = {key: tensor.to(self.device) for key, tensor in encoded.items()}
        return encoded

    def generate(self, inputs: dict[str, torch.Tensor]) -> str:
        """
        Run the model's generation pipeline and extract the SQL query.

        Args:
            inputs: Tokenized input tensors from tokenize().

        Returns:
            Generated SQL query string.
        """
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                **TEXT_TO_SQL_GENERATION_KWARGS,
            )

        generated_text = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True,
        )

        return generated_text

    def extract_sql(self, generated_text: str, original_prompt: str) -> str:
        """
        Extract the SQL query portion from the generated text.

        Removes prompt echoes, markdown formatting, and common explanatory
        text while preserving the generated SQL statements.

        Args:
            generated_text: Full text output from generate().
            original_prompt: Original user prompt for context.

        Returns:
            Extracted SQL query string.
        """

        instruction = (
            "You are a SQL expert. Generate a SQL query based on the user's request. "
            "Return ONLY the SQL query, no explanation.\n\n"
            "User request: "
        )

        # Remove instruction if model echoed it back
        if instruction in generated_text:
            idx = generated_text.find(instruction) + len(instruction)
            sql_part = generated_text[idx:].strip()
        else:
            sql_part = generated_text.strip()

        # Remove prompt echo if present
        sql_part = sql_part.replace(original_prompt, "").strip()

        # Remove markdown fences
        sql_part = sql_part.replace("```sql", "")
        sql_part = sql_part.replace("```SQL", "")
        sql_part = sql_part.replace("```", "")
        sql_part = sql_part.strip()

        cleaned_lines = []

        stop_phrases = (
            "this query",
            "the query",
            "explanation",
            "note:",
            "here is",
            "sql query:",
            "query:",
        )

        for line in sql_part.splitlines():
            line = line.strip()

            if not line:
                continue

            if line.lower().startswith(stop_phrases):
                break

            cleaned_lines.append(line)

        sql_query = " ".join(cleaned_lines).strip()

        return sql_query

    def generate_sql(self, prompt: input_prompt) -> sql_result:
        """
        Run the full pipeline to generate a SQL query from a user prompt.

        Pipeline: preprocess_prompt -> tokenize -> generate -> extract_sql

        Args:
            prompt: input_prompt containing the user's natural language request.

        Returns:
            sql_result with prompt and generated SQL query.

        Raises:
            ValueError: if the prompt is empty or invalid.
        """
        if not prompt.prompt or not prompt.prompt.strip():
            raise ValueError("Prompt cannot be empty")

        formatted = self.preprocess_prompt(prompt.prompt)
        inputs = self.tokenize(formatted)
        generated_text = self.generate(inputs)
        sql_query = self.extract_sql(generated_text, prompt.prompt)

        return sql_result(prompt=prompt.prompt, sql_query=sql_query)
