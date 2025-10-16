# src/cna_rag_agent/pipeline/rag_pipeline.py

import os
import pickle
import numpy as np
import json
import time
from typing import List, Optional, Tuple, Dict, Any
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate, BasePromptTemplate
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.language_models.chat_models import BaseChatModel
from langchain.chains import RetrievalQA

try:
    from ..config import (
        RAW_DOCUMENTS_PATH, PROCESSED_DATA_PATH, DATA_DIR,
        PREPARED_CHUNKS_CACHE_FILE, EMBEDDINGS_CACHE_FILE, CACHE_METADATA_FILE,
        RETRIEVER_K_RESULTS, RETRIEVER_SEARCH_TYPE,
        GEMINI_FLASH_MODEL_NAME, LLM_TEMPERATURE_QA, GOOGLE_API_KEY
    )
    from ..utils.logging_config import logger
    from ..data_ingestion import loader as doc_loader
    from ..data_ingestion import preprocessor as doc_preprocessor
    from ..embedding import embedder as doc_embedder
    from ..vector_store import store_manager as vs_manager
    from ..generation import generator as llm_generator
    from ..generation import prompts as llm_prompts
except ImportError:
    import sys
    SRC_DIR = Path(__file__).resolve().parent.parent.parent
    if str(SRC_DIR) not in sys.path: sys.path.insert(0, str(SRC_DIR))
    from cna_rag_agent.config import (
        RAW_DOCUMENTS_PATH, PROCESSED_DATA_PATH, DATA_DIR, PREPARED_CHUNKS_CACHE_FILE,
        EMBEDDINGS_CACHE_FILE, CACHE_METADATA_FILE, RETRIEVER_K_RESULTS,
        RETRIEVER_SEARCH_TYPE, GEMINI_FLASH_MODEL_NAME, LLM_TEMPERATURE_QA, GOOGLE_API_KEY
    )
    from cna_rag_agent.utils.logging_config import logger
    from cna_rag_agent.data_ingestion import loader as doc_loader, preprocessor as doc_preprocessor
    from cna_rag_agent.embedding import embedder as doc_embedder
    from cna_rag_agent.vector_store import store_manager as vs_manager
    from cna_rag_agent.generation import generator as llm_generator, prompts as llm_prompts

def get_source_files_metadata(documents_dir: Path) -> Dict[str, float]:
    metadata = {}
    if documents_dir.exists() and documents_dir.is_dir():
        for item in documents_dir.iterdir():
            if item.is_file() and not item.name.startswith('.') and item.suffix:
                try: metadata[item.name] = item.stat().st_mtime
                except Exception as e: logger.warning(f"Could not get stat for file {item.name}: {e}")
    manual_links_path = DATA_DIR / "manual_links.json"
    if manual_links_path.exists():
        metadata["manual_links.json"] = manual_links_path.stat().st_mtime
    return metadata

def is_cache_valid(documents_dir: Path, cache_metadata_file: Path, chunks_cache_file: Path, embeddings_cache_file: Path) -> bool:
    if not all([cache_metadata_file.exists(), chunks_cache_file.exists(), embeddings_cache_file.exists()]): return False
    try:
        with open(cache_metadata_file, 'r') as f: cached_file_info = json.load(f)
        current_file_info = get_source_files_metadata(documents_dir)
        if cached_file_info == current_file_info:
            logger.info("Cache is valid.")
            return True
        else:
            logger.info("Cache stale: Source document or manual links have changed.")
            return False
    except Exception: return False

def save_to_cache(chunks: List[Document], embeddings: List[List[float]], documents_dir: Path, cache_metadata_file: Path, chunks_cache_file: Path, embeddings_cache_file: Path):
    try:
        logger.info(f"Saving {len(chunks)} chunks to: {chunks_cache_file}")
        with open(chunks_cache_file, 'wb') as f: pickle.dump(chunks, f)
        logger.info(f"Saving {len(embeddings)} embeddings to: {embeddings_cache_file}")
        np.save(embeddings_cache_file, np.array(embeddings, dtype=object), allow_pickle=True)
        source_metadata = get_source_files_metadata(documents_dir)
        with open(cache_metadata_file, 'w') as f: json.dump(source_metadata, f, indent=4)
        logger.info("Cache saved successfully.")
    except Exception as e:
        logger.error(f"Error saving to cache: {e}", exc_info=True)

def load_from_cache(chunks_cache_file: Path, embeddings_cache_file: Path) -> Tuple[Optional[List[Document]], Optional[List[List[float]]]]:
    try:
        logger.info(f"Loading chunks from: {chunks_cache_file}")
        with open(chunks_cache_file, 'rb') as f: chunks = pickle.load(f)
        logger.info(f"Loading embeddings from: {embeddings_cache_file}")
        embeddings_array = np.load(embeddings_cache_file, allow_pickle=True)
        embeddings = [list(emb) for emb in embeddings_array]
        if len(chunks) != len(embeddings): return None, None
        logger.info(f"Loaded {len(chunks)} chunks and {len(embeddings)} embeddings from cache.")
        return chunks, embeddings
    except Exception:
        return None, None

def _check_link_with_llm(article_summary: str, paradigm_description: str) -> bool:
    prompt_template = PromptTemplate.from_template(
        """You are a research assistant. Based on the article summary and the paradigm description below, does the article concretely investigate or provide a primary example of this experimental paradigm?
        Your answer MUST be a single word: YES or NO.
        Article Summary:
        {summary_text}
        ---
        Paradigm Description:
        {paradigm_description}
        Answer (YES or NO):"""
    )
    llm = llm_generator.get_llm(model_name=GEMINI_FLASH_MODEL_NAME, temperature=0.0)
    if not llm: return False
    chain = prompt_template | llm
    try:
        response = chain.invoke({"summary_text": article_summary, "paradigm_description": paradigm_description})
        return "YES" in response.content.strip().upper()
    except Exception as e:
        logger.error(f"LLM call failed during link check: {e}", exc_info=True)
        time.sleep(2)
        return False

def run_full_ingestion_pipeline(documents_dir: Optional[Path] = None, clear_vector_store: bool = True, force_reprocess: bool = False):
    logger.info("--- Starting FINAL Corrected Ingestion & Linking Pipeline ---")
    if documents_dir is None: documents_dir = RAW_DOCUMENTS_PATH

    manual_links = {}
    manual_links_path = DATA_DIR / "manual_links.json"
    if manual_links_path.exists():
        try:
            with open(manual_links_path, 'r') as f:
                manual_links = json.load(f)
            logger.info(f"Successfully loaded {len(manual_links)} articles from manual_links.json")
        except Exception as e:
            logger.error(f"Could not load or parse manual_links.json: {e}")
    
    if not force_reprocess and is_cache_valid(documents_dir, CACHE_METADATA_FILE, PREPARED_CHUNKS_CACHE_FILE, EMBEDDINGS_CACHE_FILE):
        logger.info("Valid cache found. Loading pre-processed and linked data from cache.")
        chunks, embeddings = load_from_cache(PREPARED_CHUNKS_CACHE_FILE, EMBEDDINGS_CACHE_FILE)
    else:
        if force_reprocess: logger.info("Force reprocessing. Bypassing cache.")
        
        all_elements = doc_loader.load_all_documents(documents_dir)
        
        handbook_path_name = "BEST_Handbook of CNfA Experimental Paradigms.docx"
        handbook_elements = [e for e in all_elements if e.metadata.get("file_name") == handbook_path_name]
        
        if not handbook_elements:
            logger.error(f"CRITICAL: Handbook ('{handbook_path_name}') not found. Cannot perform linking.")
            return

        logger.info("--- Step 1a: Processing Handbook ---")
        handbook_chunks = doc_preprocessor.chunk_all_document_elements(handbook_elements)
        paradigm_defs = {chunk.metadata.get("paradigm_name"): chunk.page_content for chunk in handbook_chunks if chunk.metadata.get("paradigm_section") == "What is Studied" and "Uncategorized" not in chunk.metadata.get("paradigm_name", "") and "General Introduction" not in chunk.metadata.get("paradigm_name", "")}
        logger.info(f"Extracted {len(paradigm_defs)} paradigm definitions from the handbook.")
        
        all_final_chunks = list(handbook_chunks)
        
        article_elements_grouped = {}
        for element in all_elements:
            file_name = element.metadata.get("file_name")
            if file_name and file_name != handbook_path_name:
                article_elements_grouped.setdefault(file_name, []).append(element)

        for file_name, elements in article_elements_grouped.items():
            logger.info(f"\n--- Step 1b: Processing Article: {file_name} ---")
            article_chunks = doc_preprocessor.chunk_all_document_elements(elements)
            
            found_links = []
            if file_name in manual_links:
                found_links = manual_links.get(file_name, [])
                logger.info(f"MANUAL LINK FOUND for '{file_name}': {found_links}")
            elif paradigm_defs:
                summary = generate_and_cache_article_summary(file_name=file_name)
                if summary:
                    for name, desc in paradigm_defs.items():
                        if _check_link_with_llm(summary, desc):
                            logger.info(f"AUTO LINK FOUND: '{file_name}' -> '{name}'")
                            found_links.append(name)
                else:
                    logger.warning(f"Could not generate summary for {file_name}, skipping auto-linking.")
            
            if found_links:
                logger.info(f"Attaching {len(found_links)} links to all chunks of {file_name}")
                for chunk in article_chunks:
                    chunk.metadata['linked_paradigm'] = json.dumps(found_links)
            
            all_final_chunks.extend(article_chunks)

        logger.info(f"\n--- Step 2: Embedding all {len(all_final_chunks)} processed chunks ---")
        chunks, embeddings = doc_embedder.embed_prepared_documents(all_final_chunks)
        if not chunks:
            logger.error("Halted: Embedding process failed."); return
            
        save_to_cache(chunks, embeddings, documents_dir, CACHE_METADATA_FILE, PREPARED_CHUNKS_CACHE_FILE, EMBEDDINGS_CACHE_FILE)

    if not chunks or not embeddings:
        logger.error("Halted: No chunks or embeddings available to store."); return

    logger.info(f"--- Step 3: Storing {len(chunks)} final chunks in the vector store ---")
    if vs_manager.add_documents_to_store(documents=chunks, embeddings=embeddings, clear_existing_collection=clear_vector_store):
        logger.info("--- Ingestion & Linking Pipeline Complete ---")
    else:
        logger.error("--- Ingestion & Linking Pipeline Failed ---")

_qa_pipeline_cache: Dict[Tuple, RetrievalQA] = {}
def setup_qa_chain(user_query: Optional[str] = None, k_articles_for_selection: int = 3, article_filter_categories: List[str] = ["Title", "Abstract", "Introduction", "Header", "NarrativeText"], retriever_k_results: int = RETRIEVER_K_RESULTS, retriever_search_type: str = RETRIEVER_SEARCH_TYPE, llm_model_name: str = GEMINI_FLASH_MODEL_NAME, llm_temperature: float = LLM_TEMPERATURE_QA, prompt: BasePromptTemplate = llm_prompts.QA_PROMPT, chain_type: str = "stuff") -> Optional[RetrievalQA]:
    logger.info("Setting up QA chain...")
    target_filenames: Optional[List[str]] = None
    if user_query:
        logger.info("Two-step retrieval: Selecting relevant articles.")
        target_filenames = vs_manager.get_relevant_article_filenames(query=user_query, k_articles=k_articles_for_selection, filter_categories=article_filter_categories)
        if not target_filenames: logger.warning("Two-step: No specific articles found.")
    cache_key_retriever_part = tuple(sorted(target_filenames)) if target_filenames else ("global",)
    cache_key_full = (cache_key_retriever_part, retriever_k_results, retriever_search_type, llm_model_name, llm_temperature, prompt.template)
    if cache_key_full in _qa_pipeline_cache:
        return _qa_pipeline_cache[cache_key_full]
    try:
        # <<< FIX: Corrected variable name from 'search_type' to 'retriever_search_type' to match the function parameter >>>
        retriever = vs_manager.get_retriever(k_results=retriever_k_results, search_type=retriever_search_type, target_filenames=target_filenames)
        if retriever is None: return None
        llm = llm_generator.get_llm(model_name=llm_model_name, temperature=llm_temperature)
        if llm is None: return None
        qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type=chain_type, retriever=retriever, return_source_documents=True, chain_type_kwargs={"prompt": prompt})
        logger.info(f"RetrievalQA pipeline configured: targeting: {target_filenames if target_filenames else 'all docs'}.")
        _qa_pipeline_cache[cache_key_full] = qa_chain
        return qa_chain
    except Exception as e:
        logger.error(f"Error setting up QA pipeline: {e}", exc_info=True)
        return None

def generate_and_cache_article_summary(file_name: str, documents_dir: Path = RAW_DOCUMENTS_PATH, summaries_dir: Path = PROCESSED_DATA_PATH / "summaries") -> Optional[str]:
    if not file_name: return None
    file_path = documents_dir / file_name
    summaries_dir.mkdir(parents=True, exist_ok=True)
    summary_cache_file = summaries_dir / f"{Path(file_name).stem}_3pager_summary.md"
    if summary_cache_file.exists():
        logger.info(f"Loading cached summary for '{file_name}' from: {summary_cache_file}")
        try:
            with open(summary_cache_file, 'r', encoding='utf-8') as f: return f.read()
        except Exception as e:
            logger.error(f"Error loading cached summary: {e}. Regenerating.", exc_info=True)
    if not file_path.is_file():
        logger.error(f"Article file not found: {file_path}")
        return None
    logger.info(f"No cached summary found for '{file_name}'. Generating new summary...")
    article_full_text = doc_loader.get_full_text_for_article(file_path)
    if not article_full_text: return None
    summary_text = llm_generator.generate_detailed_article_summary(article_full_text)
    if summary_text:
        try:
            with open(summary_cache_file, 'w', encoding='utf-8') as f: f.write(summary_text)
            logger.info(f"Saved new summary for '{file_name}' to: {summary_cache_file}")
        except Exception as e:
            logger.error(f"Error saving generated summary: {e}", exc_info=True)
        return summary_text
    return None