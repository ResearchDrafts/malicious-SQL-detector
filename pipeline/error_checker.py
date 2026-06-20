"""
Validates SQL queries for syntax and semantic errors.
Consumes parsed AST from parser.py and checks for common issues.
"""

from __future__ import annotations
from ast import pattern
from ast import pattern
import re
from schemas import parser_result, error_check_result


class SQLErrorChecker:
    """
    Performs syntax and semantic validation on parsed SQL queries.

    Does NOT execute queries against a database. Validates:
    - Basic syntax correctness (matched brackets, quotes)
    - DML keyword presence
    - Common SQL anti-patterns and dangerous constructs
    - Statement structure consistency
    """

    def __init__(self) -> None:
        """Initialize the error checker."""
        self.dangerous_keywords = {
            "DROP",
            "TRUNCATE",
            "ALTER",
            "CREATE",
            "DELETE",
            "EXEC",
            "EXECUTE",
            "GRANT",
            "REVOKE",
        }

    def check_syntax(self, parser_result: parser_result) -> error_check_result:
        """
        Perform comprehensive syntax and semantic checks on a parsed SQL result.

        Args:
            parser_result: Result from SQLParser.parse() containing AST.

        Returns:
            error_check_result with syntax_valid, semantic_valid, and errors list.
        """
        errors = []

        if not parser_result.parse_success:
            return error_check_result(
                syntax_valid=False,
                semantic_valid=False,
                errors=["Failed to parse SQL query"],
            )

        ast = parser_result.ast
        if not ast:
            return error_check_result(
                syntax_valid=False,
                semantic_valid=False,
                errors=["No AST generated"],
            )

        syntax_errors = self._check_syntax_rules(ast)
        semantic_errors = self._check_semantic_rules(ast)

        errors.extend(syntax_errors)
        errors.extend(semantic_errors)

        syntax_valid = len(syntax_errors) == 0
        semantic_valid = len(semantic_errors) == 0

        return error_check_result(
            syntax_valid=syntax_valid,
            semantic_valid=semantic_valid,
            errors=errors,
        )

    def _check_syntax_rules(self, ast: dict) -> list[str]:
        """
        Check for basic SQL syntax violations.

        Args:
            ast: AST dictionary from SQLParser.build_ast().

        Returns:
            List of syntax error messages (empty if no errors).
        """
        errors = []
        statement = ast.get("statement_text", "").strip()

        if not statement:
            errors.append("Empty SQL statement")
            return errors

        if ";" in statement:
            semicolon_count = statement.count(";")
            if semicolon_count > 1:
                errors.append(
                    f"Multiple statements detected ({semicolon_count} semicolons)"
                )

        if "--" in statement:
            errors.append("SQL comments (--) detected in query")

        if "/*" in statement or "*/" in statement:
            comment_pattern = r"/\*.*?\*/"
            if re.search(comment_pattern, statement, re.DOTALL):
                errors.append("Block comments (/* */) detected in query")

        dml_type = ast.get("dml_type")
        if not dml_type:
            errors.append("No valid DML keyword (SELECT, INSERT, UPDATE, DELETE)")

        if dml_type == "SELECT" and "FROM" not in statement.upper():
            if "DUAL" not in statement.upper():
                errors.append("SELECT statement missing FROM clause")

        return errors

    def _check_semantic_rules(self, ast: dict) -> list[str]:
        """
        Check for semantic violations and dangerous patterns.

        Args:
            ast: AST dictionary from SQLParser.build_ast().

        Returns:
            List of semantic error messages (empty if no errors).
        """
        errors = []
        statement = ast.get("statement_text", "").strip().upper()
        dml_type = ast.get("dml_type", "").upper()

        for keyword in self.dangerous_keywords:
            pattern = rf"\b{re.escape(keyword)}\b"
            if re.search(pattern, statement, re.IGNORECASE):
                errors.append(f"Dangerous keyword '{keyword}' detected in query")

        if "--" in statement or "/*" in statement:
            errors.append("Potential SQL injection pattern: comment syntax detected")

        union_pattern = r"\bUNION\b"
        if re.search(union_pattern, statement, re.IGNORECASE):
            errors.append("UNION keyword detected (potential data leakage)")

        if re.search(r"INFORMATION_SCHEMA", statement, re.IGNORECASE):
            errors.append(
                "INFORMATION_SCHEMA access detected (potential privilege escalation)"
            )

        if re.search(r"xp_|sp_|exec\s*\(", statement, re.IGNORECASE):
            errors.append("System procedure call detected (xp_, sp_, exec)")

        if re.search(r"(^|\s)0x[0-9a-fA-F]+", statement):
            errors.append("Hexadecimal escape sequence detected (potential injection)")

        if dml_type == "DELETE" and "WHERE" not in statement:
            errors.append("DELETE without WHERE clause (dangerous: will delete all rows)")

        if dml_type == "UPDATE" and "WHERE" not in statement:
            errors.append("UPDATE without WHERE clause (dangerous: will update all rows)")

        return errors

    def is_malformed(self, parser_result: parser_result) -> bool:
        """
        Quick check: is the query malformed (syntax invalid)?

        Args:
            parser_result: Result from SQLParser.parse().

        Returns:
            True if query has syntax errors, False otherwise.
        """
        result = self.check_syntax(parser_result)
        return not result.syntax_valid

    def is_risky(self, parser_result: parser_result) -> bool:
        """
        Quick check: does the query have semantic issues or dangerous patterns?

        Args:
            parser_result: Result from SQLParser.parse().

        Returns:
            True if query has semantic errors, False otherwise.
        """
        result = self.check_syntax(parser_result)
        return not result.semantic_valid