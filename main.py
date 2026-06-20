"""
Main orchestrator for the malicious SQL detector pipeline.

Coordinates the full workflow:
    input -> text_to_sql -> parser -> error_checker -> llm_checker ->
    codebert -> encoder -> final_classifier -> output
"""

from __future__ import annotations
import sys
from pathlib import Path
from typing import Optional
from dataclasses import asdict

from pipeline.input import InputReader
from pipeline.text_to_sql import TextToSQLGenerator
from pipeline.parser import SQLParser
from pipeline.error_checker import SQLErrorChecker
from pipeline.llm_checker import LLMChecker
from pipeline.codebert import UnixCoderEncoder
from pipeline.encoder import FeatureEncoder
from pipeline.final_classifier import FinalClassifier
from schemas import final_result, input_prompt

INPUT_CSV = "./data/raw/multilingual_sql_security_dataset.csv"


class MaliciousSQLDetectorPipeline:
    """
    End-to-end pipeline for detecting malicious SQL queries.

    Integrates all modules (text-to-SQL, parsing, validation, LLM analysis,
    feature encoding, classification) into a single orchestrator.
    """

    def __init__(self) -> None:
        """
        Initialize the pipeline by loading all model components.

        Raises:
            RuntimeError: if any component fails to load.
        """
        try:
            self.text_to_sql = TextToSQLGenerator()
            self.parser = SQLParser()
            self.error_checker = SQLErrorChecker()
            self.llm_checker = LLMChecker()
            self.codebert = UnixCoderEncoder()
            self.encoder = FeatureEncoder()
            self.classifier = FinalClassifier()
        except Exception as exc:
            raise RuntimeError(f"Failed to initialize pipeline: {exc}") from exc

    def process_prompt(self, prompt: input_prompt) -> final_result:
        """
        Run a single prompt through the entire detection pipeline.

        Pipeline stages:
            1. text_to_sql.generate_sql()         -> sql_result
            2. parser.parse()                     -> parser_result
            3. error_checker.check_syntax()       -> error_check_result
            4. llm_checker.analyze()              -> LLM_output
            5. codebert.encode()                  -> codebert_output
            6. encoder.encode()                   -> combined_feature
            7. classifier.predict()               -> prediction

        Args:
            prompt: input_prompt containing the user's request.

        Returns:
            final_result with all analysis and prediction results.

        Raises:
            ValueError: if any pipeline stage fails.
        """
        try:
            sql_result_obj = self.text_to_sql.generate_sql(prompt)

            parsed = self.parser.parse(sql_result_obj)
            if not parsed.parse_success:
                raise ValueError(f"SQL parsing failed: {sql_result_obj.sql_query}")

            error_result = self.error_checker.check_syntax(parsed)
            if not error_result.syntax_valid:
                error_msg = "; ".join(error_result.errors)
                raise ValueError(f"SQL syntax error: {error_msg}")

            llm_output = self.llm_checker.analyze(
                prompt.prompt, sql_result_obj.sql_query
            )

            codebert_output = self.codebert.encode(sql_result_obj.sql_query)

            combined_feature = self.encoder.encode(codebert_output, llm_output)

            prediction = self.classifier.predict(combined_feature)

            return final_result(
                prompt=prompt.prompt,
                sql_query=sql_result_obj.sql_query,
                ambiguous=llm_output.ambiguous,
                ambiguity_reason=llm_output.ambiguity_reason,
                malicious=llm_output.malicious or (prediction.label == 1),
                malicious_reason=llm_output.malicious_reason,
                confidence=max(prediction.confidence, llm_output.confidence / 100.0),
                prediction=prediction.label,
            )

        except Exception as exc:
            raise ValueError(
                f"Pipeline processing failed for prompt '{prompt.prompt}': {exc}"
            ) from exc

    def process_batch(
        self, prompts: list[input_prompt]
    ) -> list[final_result]:
        """
        Process a batch of prompts through the pipeline.

        Args:
            prompts: List of input_prompt objects.

        Returns:
            List of final_result objects (one per prompt).
        """
        results = []
        for i, prompt in enumerate(prompts):
            try:
                result = self.process_prompt(prompt)
                results.append(result)
            except ValueError as exc:
                print(f"Error processing prompt {i + 1}: {exc}", file=sys.stderr)
                continue

        return results

    def process_csv(self, csv_path: str | Path) -> list[final_result]:
        """
        Read prompts from a CSV file and process them through the pipeline.

        Args:
            csv_path: Path to the CSV file.

        Returns:
            List of final_result objects.

        Raises:
            FileNotFoundError: if CSV file does not exist.
            ValueError: if CSV is malformed.
        """
        reader = InputReader(csv_path)
        results = []

        try:
            for prompt in reader.read_prompts():
                try:
                    result = self.process_prompt(prompt)
                    results.append(result)
                except ValueError as exc:
                    print(f"Error processing prompt: {exc}", file=sys.stderr)
                    continue
        except Exception as exc:
            print(f"Error reading CSV: {exc}", file=sys.stderr)

        return results


def main() -> None:
    """
    Entry point for the pipeline. Accepts CSV file path as argument.

    Usage:
        python main.py [path_to_csv]

    Outputs results to stdout and errors to stderr.
    """
    csv_path = sys.argv[1] if len(sys.argv) > 1 else INPUT_CSV

    try:
        pipeline = MaliciousSQLDetectorPipeline()
        results = pipeline.process_csv(csv_path)

        for result in results:
            result_dict = asdict(result)
            print(result_dict)

    except Exception as exc:
        print(f"Fatal error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
