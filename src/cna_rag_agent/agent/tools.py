### START OF FINAL CORRECTED FILE: src/cna_rag_agent/agent/tools.py ###

import sys
import json
from pathlib import Path

# --- Setup Project Path ---
SRC_DIR = Path(__file__).resolve().parent.parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
# --- End Setup Project Path ---

from langchain.chains import RetrievalQA
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from cna_rag_agent.utils.logging_config import logger
from cna_rag_agent.vector_store.store_manager import get_vector_store
from cna_rag_agent.generation.generator import get_llm
from cna_rag_agent.pipeline.rag_pipeline import generate_and_cache_article_summary
from cna_rag_agent.config import (
    GEMINI_FLASH_MODEL_NAME, 
    GEMINI_PRO_MODEL_NAME, 
    RETRIEVER_K_RESULTS, 
    RETRIEVER_SEARCH_TYPE,
    QA_TEMPLATE
)

# <<< FIX 1: Renamed and upgraded this tool to be a powerful, general-purpose retriever >>>
def get_information_on_topic(topic_name: str) -> str:
    """
    Retrieves and summarizes information on any given topic, category, or paradigm from the local documents.
    This is a general-purpose tool for information gathering.
    """
    logger.info(f"Tool 'get_information_on_topic' called for topic: '{topic_name}'")
    vector_store = get_vector_store()
    if not vector_store:
        return "Error: Could not access the vector store."

    # Use the vector store's similarity search to find the most relevant chunks for any topic.
    retriever = vector_store.as_retriever(
        search_type=RETRIEVER_SEARCH_TYPE,
        search_kwargs={'k': 3} # Retrieve top 3 most relevant chunks
    )
    
    retrieved_docs = retriever.invoke(topic_name)

    if not retrieved_docs:
        return f"Sorry, I could not find any information for the topic: {topic_name}"

    context = "\n\n".join(doc.page_content for doc in retrieved_docs)
    prompt = PromptTemplate.from_template(
        """Based ONLY on the following context, write a concise and clear summary of the topic.
Context:
{context}
Summary:"""
    )
    llm = get_llm(model_name=GEMINI_FLASH_MODEL_NAME, temperature=0.1)
    rag_chain = ({ "context": lambda x: context } | prompt | llm | StrOutputParser())
    result = rag_chain.invoke("") 
    logger.info(f"Successfully generated summary for topic: '{topic_name}'")
    return result

def list_articles_and_summaries_for_topic(topic_name: str) -> str:
    """Lists articles and their summaries for a given topic."""
    logger.info(f"Tool 'list_articles_and_summaries_for_topic' called for topic: '{topic_name}'")
    vector_store = get_vector_store()
    if not vector_store:
        return "Error: Could not access the vector store."
    
    collection = vector_store._collection
    all_docs_results = collection.get(include=["metadatas"])
    
    if not all_docs_results or not all_docs_results['ids']:
        return "Could not retrieve any documents from the database."
        
    simple_search_term = topic_name.split('→')[0].strip().split('. ')[-1]

    linked_files = set()
    for meta in all_docs_results['metadatas']:
        linked_paradigm_str = meta.get('linked_paradigm')
        if linked_paradigm_str and simple_search_term in linked_paradigm_str:
            if 'file_name' in meta:
                linked_files.add(meta['file_name'])

    if not linked_files:
        return f"I could not find any articles specifically linked to the topic: '{topic_name}'."

    output = []
    output.append(f"Found {len(linked_files)} article(s) linked to '{topic_name}':\n")
    
    for file_name in linked_files:
        output.append(f"\n{'='*20}\nARTICLE: {file_name}\n{'='*20}\n")
        summary = generate_and_cache_article_summary(file_name=file_name)
        if summary:
            output.append(summary)
        else:
            output.append(f"Could not generate or find a summary for {file_name}.")
        output.append("\n\n")

    return "".join(output)

def deep_dive_into_methods(paradigm_name: str) -> str:
    """Provides a deep dive into the methods of a specific paradigm."""
    logger.info(f"Tool 'deep_dive_into_methods' called for paradigm: '{paradigm_name}'")
    vector_store = get_vector_store()
    if not vector_store:
        return "Error: Could not access the vector store."
        
    collection = vector_store._collection
    handbook_results = collection.get(
        where={"$and": [{"content_type": "handbook_entry"}, {"paradigm_name": paradigm_name}]},
        include=["documents", "metadatas"]
    )
    
    handbook_context = ""
    if handbook_results and handbook_results['documents']:
        sections = {}
        for doc, meta in zip(handbook_results['documents'], handbook_results['metadatas']):
            section_name = meta.get('paradigm_section', 'Details')
            sections.setdefault(section_name, []).append(doc)
        
        for section_name, contents in sorted(sections.items()):
            handbook_context += f"### From Handbook: {section_name}\n"
            handbook_context += "\n".join(contents)
            handbook_context += "\n\n"
    
    article_summaries = list_articles_and_summaries_for_topic(paradigm_name)
    
    prompt = PromptTemplate.from_template(
        """You are an expert in Cognitive Neuroscience for Architecture. Synthesize a detailed "deep dive" on the methods used in the '{paradigm_name}' paradigm. Use the provided context from the official handbook and the summaries of related research articles. Structure your answer clearly.
**Handbook Context:**
{handbook_context}
---
**Related Article Summaries:**
{article_summaries}
---
**Synthesized Deep Dive Report:**
"""
    )
    
    llm = get_llm(model_name=GEMINI_PRO_MODEL_NAME, temperature=0.1)
    chain = prompt | llm | StrOutputParser()
    
    result = chain.invoke({
        "paradigm_name": paradigm_name,
        "handbook_context": handbook_context or "No specific handbook details found.",
        "article_summaries": article_summaries if "I could not find any articles" not in article_summaries else "No linked articles found for this paradigm."
    })
    
    logger.info(f"Successfully generated deep dive for paradigm: '{paradigm_name}'")
    return result

def identify_open_questions(topic_name: str) -> str:
    """Identifies open questions for a given topic."""
    logger.info(f"Tool 'identify_open_questions' called for topic: '{topic_name}'")
    
    intro_context = get_information_on_topic(topic_name) # Uses the new, more powerful tool
    article_summaries = list_articles_and_summaries_for_topic(topic_name)

    prompt = PromptTemplate.from_template(
        """You are a scientific analyst. Based on the theoretical background of the topic and the limitations and future work mentioned in related research articles, identify and synthesize the key "open questions" and future research directions for the topic of '{topic_name}'.
Look for gaps between the theory and the findings. Synthesize explicitly mentioned limitations and suggestions for future studies into a coherent report.
**Theoretical Background (from Handbook):**
{handbook_context}
---
**Related Article Summaries (Discussion, Limitations, Future Work):**
{article_summaries}
---
**Synthesized Report on Open Questions & Future Directions:**
"""
    )

    llm = get_llm(model_name=GEMINI_PRO_MODEL_NAME, temperature=0.3)
    chain = prompt | llm | StrOutputParser()

    result = chain.invoke({
        "topic_name": topic_name,
        "handbook_context": intro_context or "No handbook context available.",
        "article_summaries": article_summaries if "I could not find any articles" not in article_summaries else "No linked articles found to analyze for limitations or future work."
    })
    
    logger.info(f"Successfully identified open questions for topic: '{topic_name}'")
    return result

# <<< FIX 2: This tool is now a powerful, self-contained RAG chain >>>
def general_purpose_qa(question: str) -> str:
    """
    Answers a general question by performing a similarity search across ALL available documents.
    This is a fallback tool for specific definitions or questions not covered by other tools.
    """
    logger.info(f"Tool 'general_purpose_qa' called with question: '{question}'")
    try:
        vector_store = get_vector_store()
        if not vector_store:
            return "Sorry, I could not access the vector store for a general search."
            
        retriever = vector_store.as_retriever(
            search_type=RETRIEVER_SEARCH_TYPE,
            search_kwargs={'k': RETRIEVER_K_RESULTS}
        )
        
        prompt = PromptTemplate(
            template=QA_TEMPLATE,
            input_variables=["context", "question"]
        )

        llm = get_llm(model_name=GEMINI_FLASH_MODEL_NAME)
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt}
        )
        
        logger.info("General purpose QA chain created for a global search.")
        response = qa_chain.invoke(question)
        answer = response.get("result", "No answer found.")
        sources = response.get("source_documents", [])
        
        if sources:
            source_names = set(doc.metadata.get('file_name', 'Unknown') for doc in sources)
            answer += f"\n\nSources: {', '.join(source_names)}"
            
        return answer
    except Exception as e:
        logger.error(f"Error during general purpose Q&A: {e}", exc_info=True)
        return "An error occurred while trying to answer the question."

### END OF FINAL CORRECTED FILE ###