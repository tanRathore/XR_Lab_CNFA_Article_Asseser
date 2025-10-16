# src/scripts/debug_metadata.py

import sys
from pathlib import Path

# --- Setup Project Path ---
SRC_DIR = Path(__file__).resolve().parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
# --- End Setup Project Path ---

from cna_rag_agent.utils.logging_config import logger
from cna_rag_agent.vector_store.store_manager import get_vector_store

def inspect_article_metadata(file_name: str):
    """
    Connects to the DB and prints the full metadata for the first chunk
    of a specific article file to diagnose linking issues.
    """
    logger.info(f"Connecting to vector store to inspect metadata for article: '{file_name}'")
    vector_store = get_vector_store()
    if not vector_store:
        logger.error("Could not connect to vector store.")
        return

    collection = vector_store._collection
    
    # Get the first document matching the file name
    results = collection.get(
        where={"file_name": file_name},
        limit=1, # We only need to see one example
        include=["metadatas"]
    )
    
    if not results or not results['ids']:
        logger.error(f"Could not find any documents for file_name='{file_name}'.")
        return

    logger.info(f"Found chunk for '{file_name}'. Inspecting its full metadata...\n")
    
    # Print the full metadata dictionary for the first chunk found
    first_metadata = results['metadatas'][0]
    
    print(f"--- Full Metadata for a chunk from: {file_name} ---")
    import json
    # Pretty print the dictionary
    print(json.dumps(first_metadata, indent=2))
    print("-" * 50)
        
if __name__ == "__main__":
    # We want to see the metadata for the article we linked
    inspect_article_metadata("EmbodiedCognition.pdf")