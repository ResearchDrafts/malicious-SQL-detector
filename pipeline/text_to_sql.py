"""Prompt-to-SQL generation using Qwen3-4B."""

from __future__ import annotations

import re

import sqlparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from config import (
    TEXT_TO_SQL_ENABLE_THINKING,
    TEXT_TO_SQL_GENERATION_KWARGS,
    TEXT_TO_SQL_LOAD_IN_4BIT,
    TEXT_TO_SQL_MAX_LENGTH,
    TEXT_TO_SQL_MODEL_NAME,
)
from schemas import input_prompt, sql_result


SYSTEM_PROMPT = (
    "You are a Text-to-SQL assistant. Convert the user's request into exactly "
    "one SQL statement. Return only the SQL statement, with no markdown, "
    "explanation, commentary, or additional statements."
)


class TextToSQLGenerator:
    """Generate one SQL statement from a natural-language prompt using Qwen."""

    def __init__(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(TEXT_TO_SQL_MODEL_NAME)

        model_kwargs = {
            "device_map": "auto",
            "torch_dtype": "auto",
            "low_cpu_mem_usage": True,
        }
        if TEXT_TO_SQL_LOAD_IN_4BIT and torch.cuda.is_available():
            compute_dtype = (
                torch.bfloat16
                if torch.cuda.is_bf16_supported()
                else torch.float16
            )
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=compute_dtype,
            )

        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                TEXT_TO_SQL_MODEL_NAME,
                **model_kwargs,
            )
            self.model.eval()
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load Text-to-SQL model "
                f"'{TEXT_TO_SQL_MODEL_NAME}': {exc}"
            ) from exc

    def preprocess_prompt(self, prompt: str) -> str:
        """Apply Qwen's chat template to a prompt."""
        if not prompt or not prompt.strip():
            return ""

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt.strip()},
        ]
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=TEXT_TO_SQL_ENABLE_THINKING,
        )

    def tokenize(self, text: str) -> dict[str, torch.Tensor]:
        """Tokenize a chat-formatted prompt on the model's input device."""
        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=TEXT_TO_SQL_MAX_LENGTH,
            return_tensors="pt",
        )
        return encoded.to(self.model.device)

    def generate(self, inputs: dict[str, torch.Tensor]) -> str:
        """Generate and decode only tokens produced after the input prompt."""
        generation_kwargs = dict(TEXT_TO_SQL_GENERATION_KWARGS)
        generation_kwargs["pad_token_id"] = self.tokenizer.eos_token_id

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                **generation_kwargs,
            )

        input_length = inputs["input_ids"].shape[1]
        output_ids = outputs[0, input_length:]
        return self.tokenizer.decode(
            output_ids,
            skip_special_tokens=True,
        ).strip()

    def extract_sql(self, generated_text: str, original_prompt: str) -> str:
        """Remove formatting and retain only the first generated statement."""
        del original_prompt

        cleaned = re.sub(
            r"<think>.*?</think>",
            "",
            generated_text,
            flags=re.DOTALL,
        ).strip()
        cleaned = re.sub(r"```(?:sql)?|```", "", cleaned, flags=re.IGNORECASE)
        cleaned = cleaned.strip()

        statements = [statement.strip() for statement in sqlparse.split(cleaned)]
        statements = [statement for statement in statements if statement]
        return statements[0] if statements else ""

    def generate_sql(self, prompt: input_prompt) -> sql_result:
        """Generate one non-empty SQL statement for an input prompt."""
        if not prompt.prompt or not prompt.prompt.strip():
            raise ValueError("Prompt cannot be empty")

        formatted_prompt = self.preprocess_prompt(prompt.prompt)
        inputs = self.tokenize(formatted_prompt)
        generated_text = self.generate(inputs)
        sql_query = self.extract_sql(generated_text, prompt.prompt)

        if not sql_query:
            raise ValueError("Qwen returned an empty SQL query")

        return sql_result(prompt=prompt.prompt, sql_query=sql_query)
