"""
Inference orchestrator for the SQL security pipeline.

This module is inference-only. Training lives under training/ and uses
the SQL queries already present in the CSV dataset.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

from config import validate_config
from models.MLP import FinalClassifier
from pipeline.codebert import UnixCoderEncoder
from pipeline.encoder import FeatureEncoder
from pipeline.error_checker import SQLErrorChecker
from pipeline.input import InputReader
from pipeline.llm_checker import LLMChecker
from pipeline.parser import SQLParser
from pipeline.text_to_sql import TextToSQLGenerator
from schemas import final_result, input_prompt


class SQLSecurityInferencePipeline:
    """
    End-to-end inference pipeline for prompt-to-SQL security detection.
    """

    def __init__(self) -> None:
        validate_config(require_classifier=True)
        self.text_to_sql = TextToSQLGenerator()
        self.parser = SQLParser()
        self.error_checker = SQLErrorChecker()
        self.llm_checker = LLMChecker()
        self.codebert = UnixCoderEncoder()
        self.encoder = FeatureEncoder()
        self.classifier = FinalClassifier()

    def process_prompt(self, prompt: input_prompt) -> final_result:
        """
        Run a single prompt through inference and return the MLP decision.

        Parser and error-checker issues are collected when possible; only
        an unrecoverable parse failure stops the query from reaching the
        feature-generation stages.
        """
        sql_result = self.text_to_sql.generate_sql(prompt)

        parsed = self.parser.parse(sql_result)
        if not parsed.parse_success:
            raise ValueError(f"Unrecoverable SQL parsing failure: {sql_result.sql_query}")

        error_result = self.error_checker.check_syntax(parsed)
        if error_result.errors:
            print(
                f"Warnings for generated SQL: {'; '.join(error_result.errors)}",
                file=sys.stderr,
            )

        llm_output = self.llm_checker.analyze(prompt.prompt, sql_result.sql_query)
        codebert_output = self.codebert.encode(sql_result.sql_query)
        combined = self.encoder.encode(codebert_output, llm_output)
        prediction = self.classifier.predict(combined)

        return final_result(
            prompt=prompt.prompt,
            sql_query=sql_result.sql_query,
            ambiguous=llm_output.ambiguous,
            ambiguity_reason=llm_output.ambiguity_reason,
            malicious=prediction.label == 1,
            malicious_reason=(
                f"MLP classifier probability: {prediction.confidence:.4f}"
            ),
            confidence=prediction.confidence,
            prediction=prediction.label,
        )

    def process_csv(self, csv_path: str | Path) -> list[final_result]:
        reader = InputReader(csv_path)
        results: list[final_result] = []

        for prompt in reader.read_prompts_test():
            try:
                results.append(self.process_prompt(prompt))
            except Exception as exc:
                print(
                    f"Error processing prompt {prompt.prompt!r}: {exc}",
                    file=sys.stderr,
                )

        return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SQL security inference.")
    parser.add_argument("csv_path", help="CSV containing prompt rows")
    args = parser.parse_args()

    pipeline = SQLSecurityInferencePipeline()
    for result in pipeline.process_csv(args.csv_path):
        print(asdict(result))


if __name__ == "__main__":
    main()
