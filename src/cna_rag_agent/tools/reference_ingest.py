# -*- coding: utf-8 -*-
"""
Reference ingestion: load downloaded PDFs, chunk, embed, and add to existing Chroma collection.
"""

from __future__ import annotations
from pathlib import Path
from typing import List
from cna_rag_agent.utils.logging_config import logger

# LangChain doc loaders
try:
    from langchain_community.document_loaders import PyPDFLoader
except Exception as e:
    PyPDFLoader = None

from cna_rag_agent.vector_store.store_manager import get_vector_store
from cna_rag_agent.vector_store.embedder import get_embedding_model  # your CustomGeminiEmbeddings

def ingest_pdfs_into_vector_store(pdf_paths: List[Path], metadata_topic: str = "") -> int:
    if not PyPDFLoader:
        logger.error("PyPDFLoader is unavailable. pip install langchain-community pypdf")
        return 0

    docs = []
    for pdf in pdf_paths:
        try:
            loader = PyPDFLoader(str(pdf))
            file_docs = loader.load()
            # Add a little metadata to keep topic grouping
            for d in file_docs:
                d.metadata = d.metadata or {}
                d.metadata["source_path"] = str(pdf)
                if metadata_topic:
                    d.metadata["topic"] = metadata_topic
            docs.extend(file_docs)
        except Exception as e:
            logger.warning(f"[ingest] failed to load {pdf}: {e}")

    if not docs:
        return 0

    # Reuse existing persistent Chroma instance
    vs = get_vector_store()
    vs.add_documents(docs)

    logger.info(f"[ingest] Added {len(docs)} chunks from {len(pdf_paths)} PDFs.")
    return len(docs)
