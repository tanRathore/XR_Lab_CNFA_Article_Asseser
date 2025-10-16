### START OF FINAL CORRECTED FILE: src/cna_rag_agent/config.py ###

import os
from dotenv import load_dotenv
from pathlib import Path
import logging # For initial check

# Determine the base directory of the project (CNFA_RAG_Agent)
BASE_DIR = Path(__file__).resolve().parent.parent.parent
ENV_PATH = BASE_DIR / ".env"

if not ENV_PATH.exists():
    print(f"WARNING: '.env' file not found at '{ENV_PATH}'.")
    print("Please create it based on '.env.example' and add your GOOGLE_API_KEY.")
load_dotenv(dotenv_path=ENV_PATH)

# --- API Keys ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# --- Data Paths ---
DATA_DIR = BASE_DIR / "data"
RAW_DOCUMENTS_PATH = DATA_DIR / "raw_documents"
PROCESSED_DATA_PATH = DATA_DIR / "processed_data"
LOG_FILE_PATH = BASE_DIR / "app.log"

# --- Vector Store Configuration ---
VECTOR_STORE_DIR = BASE_DIR / "vector_store_data"
CHROMA_PERSIST_DIR = VECTOR_STORE_DIR / "chroma_db_cna"
CHROMA_COLLECTION_NAME = "cna_articles_collection"

# --- Embedding Model Configuration ---
EMBEDDING_MODEL_NAME = "models/text-embedding-004"
EMBEDDING_TASK_TYPE_DOCUMENT = "RETRIEVAL_DOCUMENT"
EMBEDDING_TASK_TYPE_QUERY = "RETRIEVAL_QUERY"

# --- LLM Configuration (for Gemini via Generative Language API) ---
GEMINI_PRO_MODEL_NAME = "gemini-1.5-pro-latest"
GEMINI_FLASH_MODEL_NAME = "gemini-1.5-flash-latest"
LLM_TEMPERATURE_QA = 0.0
LLM_TEMPERATURE_SUMMARIZATION = 0.2
LLM_DEFAULT_MAX_OUTPUT_TOKENS = 8192

# --- Chunking Configuration ---
CHUNK_SIZE_TOKENS = 768
CHUNK_OVERLAP_TOKENS = 150
TOKENIZER_MODEL_REFERENCE = "gpt-3.5-turbo"

# --- Retrieval Configuration ---
RETRIEVER_SEARCH_TYPE = "similarity"
RETRIEVER_K_RESULTS = 5
RETRIEVER_MMR_FETCH_K = 20
RETRIEVER_MMR_LAMBDA_MULT = 0.6

# --- Logging Configuration ---
LOG_LEVEL = "INFO"

# --- Cache Configuration ---
CACHE_DIR = BASE_DIR / ".cache"
PREPARED_CHUNKS_CACHE_FILE = CACHE_DIR / "prepared_chunks.pkl"
EMBEDDINGS_CACHE_FILE = CACHE_DIR / "embedding_vectors.npy"
CACHE_METADATA_FILE = CACHE_DIR / "cache_metadata.json"


# --- <<< FIX: ADD THIS SECTION for the missing prompt template >>> ---
QA_TEMPLATE = """
You are a helpful and precise research assistant.
Use the following pieces of retrieved context to answer the question at the end.
If you don't know the answer from the context provided, just say that you cannot answer, do not try to make up an answer.
Your goal is to provide a clear and concise answer based only on the provided text.

Context:
{context}

Question:
{question}

Helpful Answer:
"""
# --- <<< END OF ADDED SECTION >>> ---


# --- Function to create directories if they don't exist ---
def create_dir_if_not_exists(dir_path: Path):
    """Creates a directory if it doesn't already exist."""
    if not dir_path.exists():
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"INFO: Created directory: {dir_path}")

_essential_dirs = [
    DATA_DIR, RAW_DOCUMENTS_PATH, PROCESSED_DATA_PATH,
    VECTOR_STORE_DIR, CHROMA_PERSIST_DIR,
    CACHE_DIR
]
for _dir in _essential_dirs:
    create_dir_if_not_exists(_dir)

if not GOOGLE_API_KEY:
    print("CRITICAL WARNING: GOOGLE_API_KEY is not set in the '.env' file or as an environment variable.")
    print("The application will not be able to connect to Google Gemini services.")

print(f"INFO: Configuration loaded. Project base directory: {BASE_DIR}")


### END OF FINAL CORRECTED FILE ###