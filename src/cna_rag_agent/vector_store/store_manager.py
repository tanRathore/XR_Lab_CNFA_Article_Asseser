# src/cna_rag_agent/vector_store/store_manager.py

from typing import List, Optional, Any
from pathlib import Path # ENSURE THIS IS IMPORTED

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_community.vectorstores import Chroma
import chromadb

# Import from our project
try:
    from ..config import (
        CHROMA_PERSIST_DIR, CHROMA_COLLECTION_NAME, GOOGLE_API_KEY,
        RETRIEVER_K_RESULTS, RETRIEVER_SEARCH_TYPE,
        RETRIEVER_MMR_FETCH_K, RETRIEVER_MMR_LAMBDA_MULT
    )
    from ..utils.logging_config import logger
    from ..embedding.embedder import get_embedding_model
except ImportError:
    import sys
    # Path is already imported above, but if it were only here, it would be fine too for fallback.
    # from pathlib import Path # Not strictly needed again if imported at top level
    SRC_DIR = Path(__file__).resolve().parent.parent.parent
    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))
    from cna_rag_agent.config import (
        CHROMA_PERSIST_DIR, CHROMA_COLLECTION_NAME, GOOGLE_API_KEY,
        RETRIEVER_K_RESULTS, RETRIEVER_SEARCH_TYPE,
        RETRIEVER_MMR_FETCH_K, RETRIEVER_MMR_LAMBDA_MULT
    )
    from cna_rag_agent.utils.logging_config import logger
    from cna_rag_agent.embedding.embedder import get_embedding_model


_persistent_client: Optional[chromadb.PersistentClient] = None

def get_persistent_client(persist_directory: Path = CHROMA_PERSIST_DIR) -> chromadb.PersistentClient:
    global _persistent_client
    if _persistent_client is None:
        logger.info(f"Initializing ChromaDB persistent client at: {persist_directory}")
        persist_directory.mkdir(parents=True, exist_ok=True)
        _persistent_client = chromadb.PersistentClient(path=str(persist_directory))
        logger.info("ChromaDB persistent client initialized.")
    return _persistent_client

def get_vector_store(
    collection_name: str = CHROMA_COLLECTION_NAME,
    persist_directory: Path = CHROMA_PERSIST_DIR
) -> Optional[Chroma]:
    try:
        client = get_persistent_client(persist_directory)
        embedding_function = get_embedding_model()
        vector_store = Chroma(
            client=client, collection_name=collection_name,
            embedding_function=embedding_function, persist_directory=str(persist_directory)
        )
        try:
            coll_obj = client.get_collection(name=collection_name)
            if coll_obj.count() == 0: logger.warning(f"Collection '{collection_name}' exists but is empty.")
            else: logger.info(f"Collection '{collection_name}' loaded with {coll_obj.count()} documents.")
        except Exception:
             logger.error(f"Collection '{collection_name}' does not exist in the database at {persist_directory}. Please run ingestion first.")
             return None
        return vector_store
    except Exception as e:
        logger.error(f"Failed to initialize Langchain Chroma vector store: {e}", exc_info=True)
        return None

def get_relevant_article_filenames(
    query: str,
    k_articles: int = 3,
    filter_categories: List[str] = ["Title", "Abstract", "Introduction"],
    collection_name: str = CHROMA_COLLECTION_NAME,
    persist_directory: Path = CHROMA_PERSIST_DIR
) -> List[str]:
    logger.info(f"Step 1 (Article Selection): Searching for relevant articles for query: '{query[:50]}...'")
    vector_store = get_vector_store(collection_name, persist_directory)
    if not vector_store:
        logger.error("Cannot get relevant article filenames: vector store not available.")
        return []
    search_filter = None
    if filter_categories:
        search_filter = {"category": {"$in": filter_categories}}
    logger.debug(f"Article selection using filter: {search_filter}")
    try:
        retrieved_elements = vector_store.similarity_search(query, k=k_articles * 5, filter=search_filter)
    except Exception as e:
        logger.error(f"Error during similarity search for article selection: {e}", exc_info=True); return []
    if not retrieved_elements: logger.info("No relevant title/abstract elements found for article selection."); return []
    relevant_filenames = []; seen_filenames = set()
    for doc in retrieved_elements:
        filename = doc.metadata.get("file_name")
        if filename and filename not in seen_filenames:
            relevant_filenames.append(filename); seen_filenames.add(filename)
            if len(relevant_filenames) >= k_articles: break
    logger.info(f"Step 1 (Article Selection): Found {len(relevant_filenames)} relevant article filenames: {relevant_filenames}")
    return relevant_filenames

def get_retriever(
    k_results: int = RETRIEVER_K_RESULTS,
    search_type: str = RETRIEVER_SEARCH_TYPE,
    target_filenames: Optional[List[str]] = None
) -> Optional[VectorStoreRetriever]:
    action = "global" if target_filenames is None else f"targeted ({len(target_filenames)} files)"
    logger.info(f"Attempting to create {action} retriever. Search type: '{search_type}', k: {k_results}")
    vector_store = get_vector_store()
    if vector_store:
        search_kwargs: Dict[str, Any] = {"k": k_results}
        if target_filenames:
            if not isinstance(target_filenames, list) or not all(isinstance(fn, str) for fn in target_filenames):
                logger.error("Invalid target_filenames provided; must be a list of strings."); return None
            search_kwargs["filter"] = {"file_name": {"$in": target_filenames}}
            logger.info(f"Retriever will be filtered for filenames: {target_filenames}")
        if search_type == "mmr":
            search_kwargs["fetch_k"] = RETRIEVER_MMR_FETCH_K
            search_kwargs["lambda_mult"] = RETRIEVER_MMR_LAMBDA_MULT
        try:
            retriever = vector_store.as_retriever(search_type=search_type, search_kwargs=search_kwargs)
            logger.info(f"Successfully created {action} retriever with search_type='{search_type}', k={k_results}.")
            return retriever
        except Exception as e: logger.error(f"Failed to create {action} retriever from vector store: {e}", exc_info=True); return None
    else: logger.error("Failed to get vector store, cannot create retriever."); return None

def _sanitize_metadata_value(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)): return value
    try: return str(value)
    except Exception as e: logger.warning(f"Could not convert metadata value '{str(value)[:100]}' to string: {e}. Using empty string."); return ""
def _sanitize_metadata(metadata: dict) -> dict:
    if metadata is None: return {}
    return {key: _sanitize_metadata_value(value) for key, value in metadata.items()}
def add_documents_to_store(documents: List[Document], embeddings: List[List[float]], collection_name: str = CHROMA_COLLECTION_NAME, persist_directory: Path = CHROMA_PERSIST_DIR, clear_existing_collection: bool = False) -> bool:
    if not documents or not embeddings: logger.warning("No documents or embeddings provided. Skipping."); return False
    if len(documents) != len(embeddings): logger.error(f"Mismatch docs/embeddings. Cannot add."); return False
    logger.info(f"Adding {len(documents)} docs to '{collection_name}'. Clear: {clear_existing_collection}")
    try:
        client = get_persistent_client(persist_directory)
        if clear_existing_collection:
            try: logger.info(f"Deleting collection: '{collection_name}'"); client.delete_collection(name=collection_name); logger.info(f"Deleted collection: '{collection_name}'.")
            except ValueError: logger.info(f"Collection '{collection_name}' not found for deletion.")
            except Exception as e: logger.error(f"Error deleting collection '{collection_name}': {e}", exc_info=True)
        collection = client.get_or_create_collection(name=collection_name)
        doc_texts = [doc.page_content for doc in documents]
        doc_metadatas = [_sanitize_metadata(doc.metadata) for doc in documents]
        doc_ids = [f"chunk_{i}_{Path(doc.metadata.get('file_name', 'unknown')).stem}_pg{str(doc.metadata.get('page_number', 'na'))}_{str(abs(hash(doc.page_content)))[:10]}" for i, doc in enumerate(documents)]
        batch_size = 200; num_batches = (len(documents) + batch_size - 1) // batch_size
        for i in range(num_batches):
            start_idx = i * batch_size; end_idx = min((i + 1) * batch_size, len(documents))
            logger.info(f"Adding batch {i+1}/{num_batches} to Chroma '{collection_name}' ({len(doc_texts[start_idx:end_idx])} docs).")
            collection.add(embeddings=embeddings[start_idx:end_idx], documents=doc_texts[start_idx:end_idx], metadatas=doc_metadatas[start_idx:end_idx], ids=doc_ids[start_idx:end_idx])
        logger.info(f"Added/updated {len(documents)} docs in '{collection_name}'. Total in collection: {collection.count()}")
        return True
    except Exception as e: logger.error(f"Failed to add docs to Chroma '{collection_name}': {e}", exc_info=True); return False

if __name__ == "__main__":
    logger.warning("Directly running store_manager.py. Test block attempts to use embeddings and query.")
    if GOOGLE_API_KEY:
        logger.info("Running store_manager.py direct test...")
        vs_instance = get_vector_store()
        if vs_instance and vs_instance._collection.count() > 0:
            logger.info(f"Vector store '{CHROMA_COLLECTION_NAME}' loaded with {vs_instance._collection.count()} documents.")
            test_query = "embodied cognition"
            logger.info(f"\nTesting get_relevant_article_filenames for query: '{test_query}'")
            article_filenames = get_relevant_article_filenames(query=test_query, k_articles=2)
            if article_filenames: logger.info(f"Found relevant article filenames: {article_filenames}")
            else: logger.warning("No relevant article filenames found for the test query.")
            logger.info(f"\nTesting global get_retriever for query: '{test_query}'")
            global_retriever = get_retriever(k_results=3)
            if global_retriever:
                retrieved_docs_global = global_retriever.invoke(test_query)
                logger.info(f"Global retriever found {len(retrieved_docs_global)} documents:")
                for i, doc_result in enumerate(retrieved_docs_global): logger.info(f"  G-Result {i+1}: '{doc_result.page_content[:70]}...' Metadata: {doc_result.metadata.get('file_name')}, {doc_result.metadata.get('category')}")
            else: logger.error("Failed to create global retriever.")
        else: logger.warning("Vector store is empty/not loaded. Run ingestion (`python src/main.py ingest`) first.")
    else: logger.error("Direct test skipped: GOOGLE_API_KEY not set.")
    logger.info("Store_manager.py direct test finished.")