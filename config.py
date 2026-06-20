'''
for llm_checker.py
'''
import torch

# This is to prevent very large prompts or queries from being passed
max_char_limit = 2048
QWEN_MODEL_NAME = "Qwen/Qwen3-4B"
QWEN_MAX_NEW_TOKENS = 512
QWEN_ENABLE_THINKING = False
# Four-bit loading keeps Qwen practical on common Colab GPUs. It is
# automatically disabled when CUDA is unavailable.
QWEN_LOAD_IN_4BIT = True
QWEN_GENERATION_KWARGS = {
    "do_sample": False,
}

'''
for codebert.py
'''
UNIXCODER_PATH="./artifacts/unixcoder"
DEVICE="cuda" if torch.cuda.is_available() else "cpu"
MAX_SQL_LENGTH=512

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


def validate_config(*, require_classifier: bool = False) -> None:
    """
    Validate configuration values needed by the model pipelines.

    Args:
        require_classifier: When True, also require the saved final
            classifier checkpoint to exist. Training can set this to
            False because it creates that checkpoint.
    """
    from pathlib import Path

    if not DEVICE:
        raise ValueError("DEVICE must be set, for example 'cpu' or 'cuda'.")

    if not UNIXCODER_PATH:
        raise ValueError("UNIXCODER_PATH must point to the UniXcoder model directory.")
    if not Path(UNIXCODER_PATH).exists():
        raise ValueError(f"UNIXCODER_PATH does not exist: {UNIXCODER_PATH}")

    if not isinstance(MAX_SQL_LENGTH, int) or MAX_SQL_LENGTH <= 0:
        raise ValueError("MAX_SQL_LENGTH must be a positive integer.")

    if not QWEN_MODEL_NAME:
        raise ValueError("QWEN_MODEL_NAME must identify a Hugging Face model.")
    if not isinstance(QWEN_MAX_NEW_TOKENS, int) or QWEN_MAX_NEW_TOKENS <= 0:
        raise ValueError("QWEN_MAX_NEW_TOKENS must be a positive integer.")

    if require_classifier:
        if not FINAL_CLASSIFIER_PATH:
            raise ValueError("FINAL_CLASSIFIER_PATH must point to trained MLP weights.")
        if not Path(FINAL_CLASSIFIER_PATH).exists():
            raise ValueError(
                f"FINAL_CLASSIFIER_PATH does not exist: {FINAL_CLASSIFIER_PATH}"
            )
