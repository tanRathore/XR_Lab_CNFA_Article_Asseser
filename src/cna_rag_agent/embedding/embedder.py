

import time
import random
from typing import List, Optional, Any # Ensure Any is imported

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings # Import base class
import google.generativeai as genai

# Import from our project
try:
    from ..config import GOOGLE_API_KEY, EMBEDDING_MODEL_NAME
    from ..config import EMBEDDING_TASK_TYPE_DOCUMENT, EMBEDDING_TASK_TYPE_QUERY # Ensure both are in config
    from ..utils.logging_config import logger
except ImportError:
    import sys
    from pathlib import Path
    SRC_DIR = Path(__file__).resolve().parent.parent.parent
    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))
    from cna_rag_agent.config import GOOGLE_API_KEY, EMBEDDING_MODEL_NAME
    from cna_rag_agent.config import EMBEDDING_TASK_TYPE_DOCUMENT, EMBEDDING_TASK_TYPE_QUERY
    from cna_rag_agent.utils.logging_config import logger

if GOOGLE_API_KEY:
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        logger.info("Google Generative AI client (for direct calls) configured successfully in embedder.")
    except Exception as e:
        logger.error(f"Failed to configure Google Generative AI client (for direct calls) in embedder: {e}", exc_info=True)
else:
    logger.error("CRITICAL in embedder: GOOGLE_API_KEY not found. Embedding will fail.")


def get_embedding_with_backoff(
    text: str,
    model_name: str = EMBEDDING_MODEL_NAME,
    task_type_str: str = EMBEDDING_TASK_TYPE_DOCUMENT,
    max_retries: int = 7,
    base_delay_seconds: float = 4.0
) -> Optional[List[float]]:
    if not GOOGLE_API_KEY:
        logger.error("Cannot generate embedding: GOOGLE_API_KEY is not configured.")
        return None
    if not text or not text.strip():
        logger.debug("Empty or whitespace-only text provided to get_embedding_with_backoff, returning None.")
        return None
    effective_task_type = task_type_str.upper()
    for attempt in range(max_retries):
        try:
            result = genai.embed_content(
                model=model_name, content=text, task_type=effective_task_type
            )
            logger.debug(f"API call for text '{text[:50]}...' returned type: {type(result)}")
            if isinstance(result, dict):
                embedding_list = result.get('embedding')
                if isinstance(embedding_list, list): return embedding_list
                else: raise Exception(f"genai.embed_content returned a dict, but 'embedding' key was not a list or was None. Value: {embedding_list}")
            elif hasattr(result, 'embedding') and result.embedding and \
                 hasattr(result.embedding, 'values') and isinstance(result.embedding.values, list):
                return result.embedding.values
            else: raise Exception(f"Unexpected API response structure from genai.embed_content. Type: {type(result)}, Result: {str(result)[:200]}")
        except Exception as e:
            error_type_name = type(e).__name__; error_message_snippet = str(e)[:200]
            if attempt < max_retries - 1:
                wait_time = base_delay_seconds * (2 ** attempt) + random.uniform(0.1, 1.0)
                logger.warning(f"Embedding attempt {attempt + 1}/{max_retries} failed for text '{text[:60]}...': {error_type_name} - {error_message_snippet}. Retrying in {wait_time:.2f} seconds...")
                time.sleep(wait_time)
            else: logger.error(f"Embedding attempt {attempt + 1}/{max_retries} (final) failed for text '{text[:60]}...': {error_type_name} - {error_message_snippet}.")
    logger.error(f"All {max_retries} retries failed for text: '{text[:60]}...'"); return None

class CustomGeminiEmbeddings(Embeddings):
    """
    Custom Langchain Embeddings class that uses our robust get_embedding_with_backoff
    for embedding documents and queries via direct genai.embed_content calls.
    """
    # These fields will be initialized by Pydantic from class variables/defaults
    # or can be overridden if an __init__ that accepts them is defined and called.
    # Since they use config values directly, no __init__ is strictly needed if no other logic is required at init.
    model_name: str = EMBEDDING_MODEL_NAME
    task_type_document: str = EMBEDDING_TASK_TYPE_DOCUMENT
    task_type_query: str = EMBEDDING_TASK_TYPE_QUERY
    # Add any other Pydantic fields if needed, e.g., client, api_key, etc.
    # However, for this setup, get_embedding_with_backoff is self-contained.

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        logger.info(f"CustomGeminiEmbeddings: Embedding {len(texts)} documents using model '{self.model_name}'.")
        embeddings_list: List[List[float]] = [] # Ensure it's initialized for the case of all failures
        
        # Using tqdm for progress if many documents (optional, can be removed if not desired here)
        # from tqdm import tqdm
        # for text_item in tqdm(texts, desc="Embedding documents (custom)", unit="doc"):
        for text_item in texts:
            emb = get_embedding_with_backoff(
                text=text_item,
                model_name=self.model_name,
                task_type_str=self.task_type_document # Use the instance's task_type_document
            )
            if emb is None:
                logger.warning(f"Failed to embed document text: '{text_item[:70]}...'. Using zero vector.")
                embeddings_list.append([0.0] * 768) # Default to zero vector of correct dimensionality
            else:
                embeddings_list.append(emb)
        return embeddings_list

    def embed_query(self, text: str) -> List[float]:
        logger.info(f"CustomGeminiEmbeddings: Embedding query '{text[:70]}...' using model '{self.model_name}'.")
        emb = get_embedding_with_backoff(
            text=text,
            model_name=self.model_name,
            task_type_str=self.task_type_query # Use the instance's task_type_query
        )
        if emb is None:
            logger.error(f"Failed to embed query: '{text[:70]}...'. Returning zero vector.")
            return [0.0] * 768 # Default to zero vector of correct dimensionality
        return emb

_custom_embedding_model_instance: Optional[CustomGeminiEmbeddings] = None

def get_embedding_model() -> CustomGeminiEmbeddings: # Return our custom class
    """
    Initializes and returns an instance of our CustomGeminiEmbeddings class.
    Relies on class variables for configuration.
    """
    global _custom_embedding_model_instance
    if _custom_embedding_model_instance is None:
        if not GOOGLE_API_KEY:
            logger.error("GOOGLE_API_KEY not found. Cannot initialize custom embedding model.")
            raise ValueError("GOOGLE_API_KEY is not set for custom embedding model.")
        try:
            logger.info(f"Initializing CustomGeminiEmbeddings (will use class defaults from config: model={EMBEDDING_MODEL_NAME})")
            # Instantiate WITHOUT passing arguments; Pydantic will use class variable defaults
            _custom_embedding_model_instance = CustomGeminiEmbeddings()
            logger.info("CustomGeminiEmbeddings model initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize CustomGeminiEmbeddings: {e}", exc_info=True)
            raise
    return _custom_embedding_model_instance


def embed_prepared_documents(prepared_chunks: List[Document]) -> tuple[List[Document], Optional[List[List[float]]]]:
    if not prepared_chunks: logger.warning("No prepared chunks provided for embedding."); return [], None
    if not GOOGLE_API_KEY: logger.error("Cannot embed documents: GOOGLE_API_KEY is not configured."); return prepared_chunks, None
    logger.info(f"Starting embedding process for {len(prepared_chunks)} prepared chunks using custom backoff (direct genai call).")
    successfully_embedded_chunks: List[Document] = []
    corresponding_embedding_vectors: List[List[float]] = []
    from tqdm import tqdm
    for i, chunk in enumerate(tqdm(prepared_chunks, desc="Embedding Chunks", unit="chunk")):
        embedding_vector = get_embedding_with_backoff(
            text=chunk.page_content, model_name=EMBEDDING_MODEL_NAME, task_type_str=EMBEDDING_TASK_TYPE_DOCUMENT
        )
        if embedding_vector is not None:
            successfully_embedded_chunks.append(chunk); corresponding_embedding_vectors.append(embedding_vector)
        else: logger.warning(f"Failed to embed chunk {i+1}/{len(prepared_chunks)} (source: {chunk.metadata.get('file_name', 'N/A')}, content: '{chunk.page_content[:50]}...') Will be excluded.")
    failed_count = len(prepared_chunks) - len(successfully_embedded_chunks)
    logger.info(f"Embedding process completed. Successfully embedded: {len(successfully_embedded_chunks)}/{len(prepared_chunks)} chunks.")
    if failed_count > 0: logger.warning(f"{failed_count} chunks could not be embedded after all retries and were excluded.")
    if not successfully_embedded_chunks: logger.error("Embedding process resulted in no successfully embedded chunks."); return [], None
    return successfully_embedded_chunks, corresponding_embedding_vectors


if __name__ == "__main__":
    logger.info("Running embedder.py directly for testing...")
    if not GOOGLE_API_KEY:
        logger.error("CRITICAL: GOOGLE_API_KEY not found. Please set it in .env for this test.")
    else:
        try:
            custom_embed_model = get_embedding_model()
            logger.info("Testing CustomGeminiEmbeddings.embed_query()...")
            query_embedding = custom_embed_model.embed_query("What is the future of AI research?")
            if query_embedding and len(query_embedding) == 768:
                logger.info(f"CustomGeminiEmbeddings.embed_query() test successful. Dim: {len(query_embedding)}. First 3: {query_embedding[:3]}")
            else:
                logger.error(f"CustomGeminiEmbeddings.embed_query() test failed or returned unexpected result. Embedding: {query_embedding}")

            logger.info("Testing CustomGeminiEmbeddings.embed_documents()...")
            doc_texts_to_embed = ["Test document 1 for custom embedder.", "Another test document for the same."]
            doc_embeddings = custom_embed_model.embed_documents(doc_texts_to_embed)
            if doc_embeddings and len(doc_embeddings) == len(doc_texts_to_embed) and \
               (len(doc_embeddings[0]) == 768 if doc_embeddings[0] else False) :
                logger.info(f"CustomGeminiEmbeddings.embed_documents() test successful for {len(doc_embeddings)} docs.")
            else:
                logger.error(f"CustomGeminiEmbeddings.embed_documents() test failed or returned unexpected result. Embeddings: {doc_embeddings}")
        except Exception as e:
            logger.error(f"Error testing CustomGeminiEmbeddings: {e}", exc_info=True)
    logger.info("Embedder.py direct test finished.")