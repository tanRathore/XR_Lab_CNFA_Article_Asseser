### START OF FINAL CORRECTED FILE: src/cna_rag_agent/generation/generator.py ###

import asyncio
import nest_asyncio

nest_asyncio.apply()

from typing import Optional, List
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import BasePromptTemplate

# Import from our project
try:
    from ..config import (
        GOOGLE_API_KEY, GEMINI_FLASH_MODEL_NAME, GEMINI_PRO_MODEL_NAME,
        LLM_TEMPERATURE_QA, LLM_TEMPERATURE_SUMMARIZATION,
        LLM_DEFAULT_MAX_OUTPUT_TOKENS
    )
    from ..utils.logging_config import logger
    from .prompts import THREE_PAGER_SUMMARY_PROMPT
except ImportError:
    import sys
    from pathlib import Path
    SRC_DIR = Path(__file__).resolve().parent.parent.parent
    if str(SRC_DIR) not in sys.path: sys.path.insert(0, str(SRC_DIR))
    from cna_rag_agent.config import (
        GOOGLE_API_KEY, GEMINI_FLASH_MODEL_NAME, GEMINI_PRO_MODEL_NAME,
        LLM_TEMPERATURE_QA, LLM_TEMPERATURE_SUMMARIZATION,
        LLM_DEFAULT_MAX_OUTPUT_TOKENS
    )
    from cna_rag_agent.utils.logging_config import logger
    from cna_rag_agent.generation.prompts import THREE_PAGER_SUMMARY_PROMPT

# Define safety settings to be less restrictive for academic content
from google.generativeai.types.safety_types import HarmCategory, HarmBlockThreshold

SAFETY_SETTINGS = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
}

_llm_instance_cache = {}

def get_llm(
    model_name: str = GEMINI_FLASH_MODEL_NAME,
    temperature: float = LLM_TEMPERATURE_QA,
    max_output_tokens: Optional[int] = None
) -> Optional[BaseChatModel]:
    cache_key = (model_name, temperature, max_output_tokens)
    if cache_key in _llm_instance_cache:
        logger.debug(f"Returning cached LLM for: {cache_key}")
        return _llm_instance_cache[cache_key]
    if not GOOGLE_API_KEY:
        logger.error("GOOGLE_API_KEY not found. Cannot init LLM.")
        return None
    final_max_tokens = max_output_tokens if max_output_tokens is not None else LLM_DEFAULT_MAX_OUTPUT_TOKENS
    try:
        logger.info(f"Initializing ChatGoogleGenerativeAI LLM: Model='{model_name}', Temp={temperature}, MaxTokens={final_max_tokens}")
        # Pass the safety_settings parameter during initialization
        llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=GOOGLE_API_KEY,
            temperature=temperature,
            max_output_tokens=final_max_tokens,
            convert_system_message_to_human=True,
            safety_settings=SAFETY_SETTINGS
        )
        logger.info("ChatGoogleGenerativeAI LLM initialized successfully.")
        _llm_instance_cache[cache_key] = llm
        return llm
    except Exception as e:
        logger.error(f"Failed to init LLM '{model_name}': {e}", exc_info=True)
        return None

def generate_detailed_article_summary(
    article_full_text: str,
    prompt_template: BasePromptTemplate = THREE_PAGER_SUMMARY_PROMPT,
    llm_model_name: str = GEMINI_PRO_MODEL_NAME,
    temperature: float = LLM_TEMPERATURE_SUMMARIZATION
) -> Optional[str]:
    if not article_full_text.strip():
        logger.warning("Cannot generate summary: article_full_text is empty.")
        return None
    
    llm = get_llm(model_name=llm_model_name, temperature=temperature)
    if not llm:
        logger.error("Cannot generate summary: Failed to initialize LLM.")
        return None
    
    logger.info(f"Generating detailed summary for article (text length: {len(article_full_text)} chars) using model {llm_model_name}...")
    
    if "article_text" not in prompt_template.input_variables:
        logger.error(f"Prompt template is missing 'article_text' input variable. Current variables: {prompt_template.input_variables}")
        return None
    
    formatted_prompt = prompt_template.format(article_text=article_full_text)
    
    try:
        response = llm.invoke(formatted_prompt)
        summary_text = response.content
        logger.info(f"Successfully generated detailed summary (length: {len(summary_text)} chars).")
        return summary_text
    except Exception as e:
        logger.error(f"Error during detailed summary generation: {e}", exc_info=True)
        return None

if __name__ == "__main__":
    logger.info("Running generator.py directly for testing LLM initialization and summarization...")
    # The rest of this test block can remain as it was.
    if not GOOGLE_API_KEY:
        logger.error("Cannot run test: GOOGLE_API_KEY not set in .env")
    else:
        qa_llm = get_llm()
        if qa_llm: logger.info(f"Successfully got QA LLM instance: {type(qa_llm)}")
        else: logger.error("Failed to get QA LLM instance for test.")
        
        logger.info("\n--- Testing Detailed Article Summarization ---")
        dummy_article = """
        Title: The Wonders of Test Data for AI Summarization
        Introduction: This document explores the critical role of well-crafted test data.
        Methodology: We employed a qualitative analysis of LLM outputs.
        Results: Model X produced coherent summaries.
        Discussion: The findings suggest that prompts must be highly specific.
        """
        summary = generate_detailed_article_summary(dummy_article)
        if summary:
            logger.info("--- Generated Dummy Article Summary ---")
            logger.info(summary)
            logger.info("------------------------------------")
        else:
            logger.error("Failed to generate summary for the dummy article in direct test.")
    logger.info("generator.py direct test finished.")

### END OF FINAL CORRECTED FILE ###