'''
Python dataclasses for clarity
'''
from dataclasses import dataclass
import numpy as np
# prompt (from database)
@dataclass
class input_prompt:
    prompt : str

# training_sample is data extracted from the Dataset
@dataclass
class training_sample:
    prompt: str
    sql_query: str
    malicious: int

# Output of text_to_sql
@dataclass
class sql_result:
    prompt: str
    sql_query: str

# Output of parser.py
@dataclass
class parser_result:
    sql_query: str
    ast: object
    parse_success: bool

# Output of error_checker.py
@dataclass
class error_check_result:
    syntax_valid: bool
    semantic_valid: bool
    errors: list[str]

# output from LLM_checker
@dataclass 
class LLM_output:
    ambiguous: bool
    ambiguity_reason: str
    malicious: bool
    malicious_reason: str
    confidence: float
    raw_response: str
    sql_adheres_prompt: bool = True
    adherence_reason: str = ""

@dataclass 
class codebert_output: 
    embedding: np.ndarray 
    probability: float

# output of final classifier
@dataclass
class prediction:
    label: int
    confidence: float

@dataclass
class llm_feature:
    vector: np.ndarray 

@dataclass
class combined_feature:
    vector: np.ndarray   

# full result of pipeline
@dataclass
class final_result:
    prompt: str
    sql_query: str
    ambiguous: bool
    ambiguity_reason: str
    malicious: bool
    malicious_reason: str
    confidence: float
    prediction: int
