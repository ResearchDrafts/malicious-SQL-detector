'''
for llm_checker.py
'''
# This is to prevent very large prompts or queries from being passed
max_char_limit = 2048
ollama_url = "http://localhost:11434/api/generate"
model = "qwen3:4b"

'''
for codebert.py
'''
UNIXCODER_PATH=""
DEVICE=""
MAX_SQL_LENGTH=""

'''
for encoder.py
'''
# Dimensionality constants (documented and enforced).
UNIXCODER_EMBEDDING_DIM = 768
MINILM_EMBEDDING_DIM = 384
SCALAR_FEATURE_COUNT = 4  # probability, malicious flag, ambiguous flag, confidence
EXPECTED_FEATURE_DIM = (
    UNIXCODER_EMBEDDING_DIM + MINILM_EMBEDDING_DIM + SCALAR_FEATURE_COUNT
)  # 1156

MINILM_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
