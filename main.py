"""
Main orchestrator for the malicious SQL detector pipeline.

Coordinates the full workflow:
    input -> text_to_sql -> parser -> error_checker -> llm_checker ->
    codebert -> encoder -> final_classifier -> output
"""

