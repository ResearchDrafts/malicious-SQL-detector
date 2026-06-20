"""
Parses SQL query strings into Abstract Syntax Tree (AST) and validates
basic syntax correctness. Delegates semantic validation to error_checker.py.
"""

from __future__ import annotations
import sqlparse
from sqlparse.sql import IdentifierList, Identifier, Where, Token
from sqlparse.tokens import Keyword, DML
from schemas import sql_result, parser_result
from typing import Optional


class SQLParser:
    """
    Parses SQL queries into Abstract Syntax Tree (AST) and performs
    basic structural validation.

    Uses sqlparse library for tokenization and basic AST construction.
    Does not perform semantic validation (e.g., table/column existence).
    """

    def __init__(self) -> None:
        """Initialize the parser."""
        pass

    def tokenize(self, sql_query: str) -> list:
        """
        Tokenize a SQL query using sqlparse.

        Args:
            sql_query: Raw SQL query string.

        Returns:
            List of sqlparse Token objects.
        """
        if not sql_query or not sql_query.strip():
            return []

        parsed = sqlparse.parse(sql_query)

        all_tokens = []
        for stmt in parsed:
            all_tokens.extend(stmt.tokens)

        return all_tokens

    def validate_basic_structure(self, tokens: list) -> tuple[bool, Optional[str]]:
        """
        Validate basic SQL structure: presence of DML keyword, brackets match, etc.

        Args:
            tokens: List of sqlparse Token objects.

        Returns:
            Tuple (is_valid, error_message). error_message is None if valid.
        """
        if not tokens:
            return False, "Empty SQL query"

        has_dml = any(
            token.ttype is DML for token in tokens if not token.is_whitespace
        )
        if not has_dml:
            return False, "No DML keyword found (SELECT, INSERT, UPDATE, DELETE)"

        open_brackets = sum(1 for token in tokens if str(token) == "(")
        close_brackets = sum(1 for token in tokens if str(token) == ")")
        if open_brackets != close_brackets:
            return False, f"Bracket mismatch: {open_brackets} open, {close_brackets} close"

        open_quotes = 0
        quote_char = None
        for token in tokens:
            token_str = str(token)
            if token_str in ("'", '"'):
                if not quote_char:
                    quote_char = token_str
                    open_quotes += 1
                elif token_str == quote_char:
                    open_quotes -= 1
                    quote_char = None

        if open_quotes != 0 or quote_char is not None:
            return False, "Unmatched quotes in SQL query"

        return True, None

    def build_ast(self, tokens: list) -> dict:
        """
        Build a simple AST representation of the parsed SQL tokens.

        Returns a dict with:
            - dml_type: the DML keyword (SELECT, INSERT, UPDATE, DELETE)
            - tokens: full token list
            - statement_text: reconstructed query string

        Args:
            tokens: List of sqlparse Token objects.

        Returns:
            Dictionary representing the AST.
        """
        dml_type = None
        for token in tokens:
            if token.ttype is DML:
                dml_type = str(token).upper()
                break

        statement_text = "".join(str(token) for token in tokens)

        ast_dict = {
            "dml_type": dml_type,
            "tokens": tokens,
            "statement_text": statement_text.strip(),
        }

        return ast_dict

    def parse(self, sql_input: sql_result) -> parser_result:
        """
        Run the full parsing pipeline: tokenize -> validate -> build AST.

        Args:
            sql_input: sql_result containing prompt and SQL query.

        Returns:
            parser_result with the AST and parse status.
        """
        tokens = self.tokenize(sql_input.sql_query)
        is_valid, error_msg = self.validate_basic_structure(tokens)

        if not is_valid:
            return parser_result(
                sql_query=sql_input.sql_query,
                ast=None,
                parse_success=False,
            )

        ast = self.build_ast(tokens)

        return parser_result(
            sql_query=sql_input.sql_query,
            ast=ast,
            parse_success=True,
        )

    def extract_tables(self, ast: dict) -> list[str]:
        """
        Extract table names from the parsed AST.

        Args:
            ast: AST dictionary from build_ast().

        Returns:
            List of table names referenced in the query.
        """
        if not ast or not ast.get("tokens"):
            return []

        tables = []
        tokens = ast["tokens"]

        from_seen = False
        join_seen = False

        for i, token in enumerate(tokens):
            token_str = str(token).upper()

            if token_str in ("FROM", "INTO", "UPDATE"):
                from_seen = True
                join_seen = False
                continue

            if token_str in ("JOIN", "LEFT JOIN", "RIGHT JOIN", "INNER JOIN"):
                join_seen = True
                continue

            if token_str in ("WHERE", "GROUP", "ORDER", "LIMIT"):
                from_seen = False
                join_seen = False
                continue

            if (from_seen or join_seen) and not token.is_whitespace:
                if isinstance(token, Identifier):
                    tables.append(str(token.get_real_name()).strip())
                elif token.ttype is None and "(" not in str(token):
                    table_name = str(token).strip()
                    if table_name and table_name not in (",", ";"):
                        tables.append(table_name)

        return tables

    def extract_columns(self, ast: dict) -> list[str]:
        """
        Extract column names from SELECT/INSERT clauses in the AST.

        Args:
            ast: AST dictionary from build_ast().

        Returns:
            List of column names referenced.
        """
        if not ast or not ast.get("tokens"):
            return []

        columns = []
        tokens = ast["tokens"]

        select_seen = False
        for token in tokens:
            token_str = str(token).upper()

            if token_str == "SELECT":
                select_seen = True
                continue

            if token_str in ("FROM", "WHERE", "GROUP", "ORDER"):
                select_seen = False
                continue

            if select_seen and isinstance(token, IdentifierList):
                for identifier in token.get_identifiers():
                    col_name = str(identifier).strip()
                    if col_name != "*":
                        columns.append(col_name)
            elif select_seen and isinstance(token, Identifier):
                col_name = str(token).strip()
                if col_name != "*":
                    columns.append(col_name)

        return columns