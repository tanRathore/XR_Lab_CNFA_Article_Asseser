# src/scripts/link_articles.py

import sys
import json
from pathlib import Path
import time

# --- Setup Project Path ---
SRC_DIR = Path(__file__).resolve().parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
# --- End Setup Project Path ---

from cna_rag_agent.utils.logging_config import logger
from cna_rag_agent.vector_store.store_manager import get_vector_store, get_persistent_client, CHROMA_COLLECTION_NAME
from cna_rag_agent.pipeline.rag_pipeline import generate_and_cache_article_summary
from cna_rag_agent.generation.generator import get_llm
# --- THIS LINE IS THE ONLY CHANGE ---
from cna_rag_agent.config import RAW_DOCUMENTS_PATH, GEMINI_FLASH_MODEL_NAME, DATA_DIR

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate


def get_all_paradigms_from_db(vector_store) -> list[dict]:
    logger.info("Fetching all unique paradigm definitions from the vector store...")
    results = vector_store._collection.get(where={"content_type": "handbook_entry"}, include=["metadatas"])
    
    paradigms = {}
    if results and results['metadatas']:
        for meta in results['metadatas']:
            paradigm_name = meta.get('paradigm_name')
            if paradigm_name and "Uncategorized" not in paradigm_name and "General Introduction" not in paradigm_name:
                description_chunk = vector_store._collection.get(
                    where={"$and": [{"paradigm_name": {"$eq": paradigm_name}}, {"paradigm_section": {"$eq": "What is Studied"}}]},
                    include=["documents"]
                )
                if description_chunk['documents']:
                    paradigms[paradigm_name] = description_chunk['documents'][0]

    logger.info(f"Found {len(paradigms)} unique paradigms to check for links.")
    return [{"name": name, "description": desc} for name, desc in paradigms.items()]

def get_all_articles_from_db(vector_store) -> list[dict]:
    logger.info("Fetching all unique scientific articles from the vector store...")
    results = vector_store._collection.get(include=["metadatas"])
    
    articles = {}
    if results and results['metadatas']:
        for meta in results['metadatas']:
            file_name = meta.get('file_name')
            if file_name and file_name != "BEST_Handbook of CNfA Experimental Paradigms.docx":
                articles[file_name] = True
            
    logger.info(f"Found {len(articles)} unique articles to process.")
    return [{"file_name": name} for name in articles.keys()]

def check_link_with_llm(article_summary: str, paradigm_description: str) -> bool:
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
    llm = get_llm(model_name=GEMINI_FLASH_MODEL_NAME, temperature=0.0)
    if not llm: return False
    chain = prompt_template | llm
    try:
        response = chain.invoke({"summary_text": article_summary, "paradigm_description": paradigm_description})
        return "YES" in response.content.strip().upper()
    except Exception: return False

def update_metadata_in_db(collection, file_name: str, new_paradigm_link: str):
    results = collection.get(where={"file_name": file_name}, include=["metadatas"])
    if not results['ids']: return

    chunk_ids = results['ids']
    existing_metadatas = results['metadatas']
    
    updated_metadatas = []
    for meta in existing_metadatas:
        if 'linked_paradigm' in meta:
            try:
                paradigm_list = json.loads(meta['linked_paradigm'])
                if not isinstance(paradigm_list, list): paradigm_list = [paradigm_list]
            except (json.JSONDecodeError, TypeError):
                paradigm_list = [meta['linked_paradigm']]
            if new_paradigm_link not in paradigm_list:
                paradigm_list.append(new_paradigm_link)
            meta['linked_paradigm'] = json.dumps(paradigm_list)
        else:
            meta['linked_paradigm'] = json.dumps([new_paradigm_link])
        updated_metadatas.append(meta)

    logger.info(f"Updating {len(chunk_ids)} chunks for '{file_name}' with new link: '{new_paradigm_link}'")
    collection.update(ids=chunk_ids, metadatas=updated_metadatas)

def main():
    logger.info("--- Starting Knowledge Base Linking Script ---")
    vector_store = get_vector_store()
    if not vector_store: return

    client = get_persistent_client()
    collection = client.get_collection(name=CHROMA_COLLECTION_NAME)

    paradigms = get_all_paradigms_from_db(vector_store)
    articles = get_all_articles_from_db(vector_store)

    if not paradigms or not articles:
        logger.error("No paradigms or articles found to link. Ensure both the handbook and other articles have been ingested.")
        return
        
    manual_links = {}
    manual_links_path = DATA_DIR / "manual_links.json"
    if manual_links_path.exists():
        with open(manual_links_path, 'r') as f:
            manual_links = json.load(f)

    total_links_found = 0
    for article in articles:
        file_name = article['file_name']
        logger.info(f"\n--- Processing article: {file_name} ---")
        
        links_to_add = []
        if file_name in manual_links:
            links_to_add = manual_links[file_name]
            logger.info(f"Found {len(links_to_add)} manual links for '{file_name}'")
        else:
            summary = generate_and_cache_article_summary(file_name=file_name, documents_dir=RAW_DOCUMENTS_PATH)
            if summary:
                for paradigm in paradigms:
                    if check_link_with_llm(summary, paradigm['description']):
                        logger.info(f"AUTO LINK FOUND: '{file_name}' -> '{paradigm['name']}'")
                        links_to_add.append(paradigm['name'])
                    time.sleep(2)
        
        if links_to_add:
            for link in links_to_add:
                total_links_found += 1
                update_metadata_in_db(collection=collection, file_name=file_name, new_paradigm_link=link)
            
    logger.info(f"\n--- Linking Script Finished ---")
    logger.info(f"Total links established/verified: {total_links_found}")

if __name__ == "__main__":
    main()