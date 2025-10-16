# src/cna_rag_agent/data_ingestion/preprocessor.py

from typing import List

from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tiktoken

# Primary attempt for package-based import
try:
    from ..config import CHUNK_SIZE_TOKENS, CHUNK_OVERLAP_TOKENS, TOKENIZER_MODEL_REFERENCE
    from ..utils.logging_config import logger
except ImportError:
    # Fallback for direct script execution
    import sys
    from pathlib import Path
    SRC_DIR = Path(__file__).resolve().parent.parent.parent # Should navigate up to 'src'
    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))
    from cna_rag_agent.config import CHUNK_SIZE_TOKENS, CHUNK_OVERLAP_TOKENS, TOKENIZER_MODEL_REFERENCE
    from cna_rag_agent.utils.logging_config import logger

MIN_ELEMENT_LENGTH_FOR_CHUNKING = 10

def get_tokenizer_for_reference_counting(encoding_name: str = "cl100k_base") -> tiktoken.Encoding:
    try:
        tokenizer = tiktoken.get_encoding(encoding_name)
    except KeyError:
        logger.error(f"Encoding '{encoding_name}' not found by tiktoken.get_encoding.")
        raise
    return tokenizer

def chunk_all_document_elements(loaded_elements: List[Document]) -> List[Document]:
    logger.info(f"Starting chunking process for {len(loaded_elements)} loaded document elements.")
    logger.info(f"Using TOKENIZER_MODEL_REFERENCE: '{TOKENIZER_MODEL_REFERENCE}' for RecursiveCharacterTextSplitter.")

    try:
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            model_name=TOKENIZER_MODEL_REFERENCE,
            chunk_size=CHUNK_SIZE_TOKENS,
            chunk_overlap=CHUNK_OVERLAP_TOKENS,
        )
        logger.info(f"Successfully initialized RecursiveCharacterTextSplitter with tiktoken model: '{TOKENIZER_MODEL_REFERENCE}'.")
    except Exception as e:
        logger.error(f"Failed to initialize RecursiveCharacterTextSplitter with tiktoken model '{TOKENIZER_MODEL_REFERENCE}': {e}", exc_info=True)
        logger.info("Falling back to character-based RecursiveCharacterTextSplitter.")
        char_chunk_size = CHUNK_SIZE_TOKENS * 4
        char_chunk_overlap = CHUNK_OVERLAP_TOKENS * 4
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=char_chunk_size,
            chunk_overlap=char_chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )
        logger.info(f"Using character-based splitter with chunk_size={char_chunk_size}, overlap={char_chunk_overlap}.")

    all_chunks: List[Document] = []
    elements_skipped = 0

    for i, element_doc in enumerate(loaded_elements):
        if not element_doc.page_content or len(element_doc.page_content.strip()) < MIN_ELEMENT_LENGTH_FOR_CHUNKING:
            logger.debug(f"Skipping element {i+1}/{len(loaded_elements)} from {element_doc.metadata.get('file_name', 'N/A')} due to short/empty content.")
            elements_skipped += 1
            continue
        
        try:
            chunks_from_element = text_splitter.split_documents([element_doc])
            if chunks_from_element:
                all_chunks.extend(chunks_from_element)
                logger.debug(f"Processed element {i+1}/{len(loaded_elements)}. Original length (chars): {len(element_doc.page_content)}. Generated {len(chunks_from_element)} chunk(s).")
            else:
                logger.debug(f"Element {i+1}/{len(loaded_elements)} from {element_doc.metadata.get('file_name', 'N/A')} resulted in no chunks.")
        except Exception as e:
            logger.error(f"Error splitting element {i+1} from {element_doc.metadata.get('file_name', 'N/A')}: {e}", exc_info=True)

    logger.info(f"Chunking process completed.")
    logger.info(f"Original document elements considered for chunking: {len(loaded_elements) - elements_skipped}")
    logger.info(f"Elements skipped due to initial short/empty content: {elements_skipped}")
    logger.info(f"Total chunks created: {len(all_chunks)}")
    return all_chunks


if __name__ == "__main__":
    logger.info("Running preprocessor.py directly for testing...")
    dummy_elements = [
        Document(
            page_content="This is the first element. It is a moderately long paragraph that should ideally form a single chunk if CHUNK_SIZE_TOKENS is large enough, or be split if it's too long based on token count. It talks about various interesting concepts in computational neuroscience.",
            metadata={"source": "test_doc.pdf", "file_name": "test_doc.pdf", "page_number": 1, "category": "NarrativeText"}
        ),
        Document(page_content="Short.", metadata={"source": "test_doc.pdf", "file_name": "test_doc.pdf", "page_number": 1, "category": "ListItem"}), # Test short skip
        Document(
            page_content="This is another element, also from page 1. " * 30, # Ensure significant length for splitting
            metadata={"source": "test_doc.pdf", "file_name": "test_doc.pdf", "page_number": 1, "category": "NarrativeText"}
        ),
    ]
    logger.info(f"Created {len(dummy_elements)} dummy document elements for preprocessor.py direct test.")
    processed_chunks = chunk_all_document_elements(dummy_elements)

    if processed_chunks:
        logger.info(f"--- Example of first created chunk from preprocessor.py direct test ---")
        first_chunk = processed_chunks[0]
        logger.info(f"Content snippet: {first_chunk.page_content[:200]}...")
        logger.info(f"Metadata: {first_chunk.metadata}")
        try:
            test_tokenizer = get_tokenizer_for_reference_counting()
            logger.info(f"Token count (using 'cl100k_base' for reference): {len(test_tokenizer.encode(first_chunk.page_content))}")
        except Exception as e:
            logger.warning(f"Could not get reference token count for test print: {e}")
    else:
        logger.warning("Preprocessor.py direct test: No chunks were created.")
    logger.info("Preprocessor.py direct test finished.")