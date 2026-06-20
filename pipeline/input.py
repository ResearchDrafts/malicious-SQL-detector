"""
Reads input prompts from CSV files and yields structured training_sample object (ie the prompt, query and label)
Handles CSV parsing, validation, and error handling.
"""

from __future__ import annotations
import csv
from pathlib import Path
from typing import Generator, Optional
from schemas import training_sample


class InputReader:
    """
    Reads CSV files and yields the training_sample object
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
        'query', 'Query', 'text', 'Text', etc. Matching is case-insensitive.

        Args:
            fieldnames: List of column names from CSV header.

        Returns:
            Column name containing prompts, or None if not found.
        """
        if not fieldnames:
            return None

        prompt_aliases = [
            "prompt",
            "prompt_entered",
            "query",
            "text",
            "request",
            "input",
        ]

        normalized = {f.lower(): f for f in fieldnames}
        for alias in prompt_aliases:
            if alias in normalized:
                return normalized[alias]

        return None

    def detect_sql_column(self, fieldnames: list[str]) -> Optional[str]:
        """
        Detect the SQL query column name from CSV header.

        Searches for common variations: 'sql_query', 'sql', 'query_sql',
        etc. Matching is case-insensitive.

        Args:
            fieldnames: List of column names from CSV header.

        Returns:
            Column name containing SQL queries, or None if not found.
        """
        if not fieldnames:
            return None

        sql_aliases = [
            "sql_query",
            "sql",
            "SQL query",
            "query_sql",
        ]

        normalized = {f.lower(): f for f in fieldnames}
        for alias in sql_aliases:
            if alias in normalized:
                return normalized[alias]

        return None

    def detect_label_column(self, fieldnames: list[str]) -> Optional[str]:
        """
        Detect the malicious-label column name from CSV header.

        Searches for common variations: 'malicious', 'is_malicious',
        'label', 'target', 'class', etc. Matching is case-insensitive.

        Args:
            fieldnames: List of column names from CSV header.

        Returns:
            Column name containing malicious labels, or None if not found.
        """
        if not fieldnames:
            return None

        label_aliases = [
            "Malicious or not",
            "malicious",
            "is_malicious",
            "label",
            "target",
            "class",
        ]

        normalized = {f.lower(): f for f in fieldnames}
        for alias in label_aliases:
            if alias in normalized:
                return normalized[alias]

        return None

    def read_prompts_train(self) -> Generator[training_sample, None, None]:
        """
        Read CSV file and yield training_sample objects for each row.

        Expected CSV format:
            - Header row containing a prompt column, a SQL query column,
            and a malicious-label column (see detect_prompt_column,
            detect_sql_column, and detect_label_column for accepted
            aliases; matching is case-insensitive).
            - Data rows: one (prompt, sql_query, malicious) triple per row.

        Skips rows where any required field is empty after stripping
        whitespace. Validates that all three required columns exist before
        iterating, that the malicious label can be converted to int, and
        that it is binary (0 or 1).

        Args:
            None

        Yields:
            training_sample objects (one per valid CSV row).

        Raises:
            ValueError: if the prompt, SQL, or label column cannot be
                detected, if the CSV is malformed, if a malicious label
                cannot be converted to int, or if it is not 0 or 1.
        """
        row_number = 1

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

                sql_col = self.detect_sql_column(reader.fieldnames)
                if not sql_col:
                    raise ValueError(
                        f"No SQL query column detected. Available columns: {reader.fieldnames}"
                    )

                label_col = self.detect_label_column(reader.fieldnames)
                if not label_col:
                    raise ValueError(
                        f"No malicious-label column detected. Available columns: {reader.fieldnames}"
                    )

                for row in reader:
                    row_number += 1

                    prompt_text = (row.get(prompt_col) or "").strip()
                    sql_text = (row.get(sql_col) or "").strip()
                    label_text = (row.get(label_col) or "").strip()

                    if not prompt_text or not sql_text or not label_text:
                        continue

                    try:
                        malicious_label = int(label_text)
                    except (TypeError, ValueError) as exc:
                        raise ValueError(
                            f"Invalid malicious label at row {row_number}: "
                            f"{label_text!r} could not be converted to int"
                        ) from exc

                    if malicious_label not in (0, 1):
                        raise ValueError(
                            f"Invalid malicious label at row {row_number}: "
                            f"expected 0 or 1, got {malicious_label}"
                        )

                    yield training_sample(
                        prompt=prompt_text,
                        sql_query=sql_text,
                        malicious=malicious_label,
                    )

        except csv.Error as exc:
            raise ValueError(f"CSV parsing error at row {row_number}: {exc}") from exc
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError(f"Error reading CSV file: {exc}") from exc
        
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