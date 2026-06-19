"""
Reads input prompts from CSV files and yields structured input_prompt objects.
Handles CSV parsing, validation, and error handling.
"""

from __future__ import annotations
import csv
from pathlib import Path
from typing import Generator, Optional
from schemas import input_prompt


class InputReader:
    """
    Reads CSV files and yields input_prompt objects.

    Expected CSV format:
        - Header row: 'prompt' (or similar column containing the SQL request)
        - Data rows: one prompt per row
    """

    def __init__(self, csv_path: str | Path) -> None:
        """
        Initialize the InputReader with a CSV file path.

        Args:
            csv_path: Path to the CSV file.

        Raises:
            FileNotFoundError: if the CSV file does not exist.
            ValueError: if the CSV file is not accessible.
        """
        self.csv_path = Path(csv_path)

        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")

        if not self.csv_path.is_file():
            raise ValueError(f"Path is not a file: {self.csv_path}")

    def detect_prompt_column(self, fieldnames: list[str]) -> Optional[str]:
        """
        Detect the prompt column name from CSV header.

        Searches for common variations: 'prompt', 'Prompt', 'PROMPT',
        'query', 'Query', 'text', 'Text', etc.

        Args:
            fieldnames: List of column names from CSV header.

        Returns:
            Column name containing prompts, or None if not found.
        """
        if not fieldnames:
            return None

        prompt_aliases = [
            "prompt",
            "Prompt",
            "PROMPT",
            "query",
            "Query",
            "QUERY",
            "text",
            "Text",
            "TEXT",
            "request",
            "Request",
            "REQUEST",
            "input",
            "Input",
            "INPUT",
        ]

        for alias in prompt_aliases:
            if alias in fieldnames:
                return alias

        return fieldnames[0] if fieldnames else None

    def read_prompts(self) -> Generator[input_prompt, None, None]:
        """
        Read CSV file and yield input_prompt objects for each row.

        Skips empty rows and validates that the prompt column exists.

        Args:
            None

        Yields:
            input_prompt objects (one per CSV row).

        Raises:
            ValueError: if the prompt column cannot be detected or CSV is malformed.
        """
        try:
            with open(self.csv_path, "r", encoding="utf-8") as csvfile:
                reader = csv.DictReader(csvfile)

                if not reader.fieldnames:
                    raise ValueError("CSV file has no headers")

                prompt_col = self.detect_prompt_column(reader.fieldnames)
                if not prompt_col:
                    raise ValueError(
                        f"No prompt column detected. Available columns: {reader.fieldnames}"
                    )

                row_number = 1
                for row in reader:
                    row_number += 1
                    prompt_text = row.get(prompt_col, "").strip()

                    if not prompt_text:
                        continue

                    yield input_prompt(prompt=prompt_text)

        except csv.Error as exc:
            raise ValueError(f"CSV parsing error at row {row_number}: {exc}") from exc
        except Exception as exc:
            raise ValueError(f"Error reading CSV file: {exc}") from exc

    def read_prompts_from_list(
        self, prompts: list[str]
    ) -> Generator[input_prompt, None, None]:
        """
        Yield input_prompt objects from a list of prompt strings.

        Useful for in-memory test data without reading from CSV.

        Args:
            prompts: List of prompt strings.

        Yields:
            input_prompt objects.

        Raises:
            ValueError: if the list is empty.
        """
        if not prompts:
            raise ValueError("Prompts list cannot be empty")

        for prompt_text in prompts:
            if isinstance(prompt_text, str) and prompt_text.strip():
                yield input_prompt(prompt=prompt_text.strip())

    def count_rows(self) -> int:
        """
        Count the total number of valid (non-empty) rows in the CSV.

        Args:
            None

        Returns:
            Number of data rows (excluding header).

        Raises:
            ValueError: if CSV cannot be read.
        """
        try:
            count = 0
            with open(self.csv_path, "r", encoding="utf-8") as csvfile:
                reader = csv.DictReader(csvfile)

                if not reader.fieldnames:
                    raise ValueError("CSV file has no headers")

                prompt_col = self.detect_prompt_column(reader.fieldnames)
                if not prompt_col:
                    raise ValueError("No prompt column detected")

                for row in reader:
                    prompt_text = row.get(prompt_col, "").strip()
                    if prompt_text:
                        count += 1

            return count
        except Exception as exc:
            raise ValueError(f"Error counting CSV rows: {exc}") from exc