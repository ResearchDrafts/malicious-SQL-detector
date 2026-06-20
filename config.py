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
EXPECTED_FEATURE_DIM = (
    UNIXCODER_EMBEDDING_DIM + MINILM_EMBEDDING_DIM
)  # 1152

MINILM_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
'''
for text_to_sql.py
'''
LLAMA_BASE_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
LLAMA_ADAPTER_PATH = "./artifacts/llama_adapter"
TEXT_TO_SQL_MAX_LENGTH = 512
TEXT_TO_SQL_GENERATION_KWARGS = {
    "max_new_tokens": 256,
    "temperature": 0.1,
    "top_p": 0.95,
    "do_sample": True,
}

'''
for final_classifier.py
'''
FINAL_CLASSIFIER_PATH = "./artifacts/classifier_model.pth"

'''
Data paths
'''
DATA_RAW_DIR = "./data/raw"
DATA_PROCESSED_DIR = "./data/processed"
INPUT_CSV = "./data/raw/multilingual_sql_security_dataset.csv"
TRAIN_CSV = "./data/processed/security_train.csv"
VAL_CSV = "./data/processed/security_val.csv"
TEST_CSV = "./data/processed/security_test.csv"
