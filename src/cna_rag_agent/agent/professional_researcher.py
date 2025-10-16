# -*- coding: utf-8 -*-
# src/cna_rag_agent/agent/professional_researcher.py

import sys
from pathlib import Path

# --- Setup Project Path ---
SRC_DIR = Path(__file__).resolve().parent.parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# --- Optional: auto-load .env ---
try:
    from dotenv import load_dotenv  # pip install python-dotenv
    load_dotenv()
except Exception:
    pass

from langchain.agents import AgentExecutor, Tool, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate

# Prefer new Tavily package; fall back to community
try:
    from langchain_tavily import TavilySearchResults  # pip install -U langchain-tavily
except Exception:  # pragma: no cover
    from langchain_community.tools.tavily_search import TavilySearchResults

from cna_rag_agent.utils.logging_config import logger
from cna_rag_agent.generation.generator import get_llm
from cna_rag_agent.config import GEMINI_PRO_MODEL_NAME
from cna_rag_agent.vector_store.store_manager import get_retriever


def create_professional_researcher_agent() -> AgentExecutor:
    """
    Gold-standard research agent that:
      1) searches local KB + web,
      2) synthesizes STRICT JSON matching our schema,
      3) returns only that JSON (no extra prose).
    """
    logger.info("Creating the 'Gold Standard' Professional Researcher AGENT...")

    # --- Tools ---
    local_retriever = get_retriever(k_results=12)
    local_search_tool = Tool(
        name="local_knowledge_base_search",
        func=lambda q: [doc.page_content for doc in local_retriever.invoke(q)],
        description=(
            "Search the curated local knowledge base of CNfA papers/handbook. "
            "Best for definitions, paradigms, and canonical methods."
        ),
    )

    web_search_tool = TavilySearchResults(max_results=15, name="online_web_search")
    web_search_tool.description = (
        "Find recent, up-to-date scholarly sources (foundational + contemporary). "
        "Use repeatedly with diverse queries until coverage is rich."
    )

    tools = [local_search_tool, web_search_tool]

    # --- System prompt with strict JSON contract ---
    system_prompt = """
You are a Distinguished Professor in Cognitive Neuroscience for Architecture (CNfA).
Task: produce a single FINAL JSON object for a 3,000-word literature review.

Method:
1) Use tools heavily first (local_knowledge_base_search, online_web_search) to gather 24–30 high-quality sources, with ≥8 published in or after 2015, and include classic foundations (Tolman/Lynch/Siegel & White).
2) STOP using tools. Then synthesize a report **strictly** in the JSON schema below. Return ONLY valid JSON (no extra commentary, no code fences).

JSON schema (keys and types must match):
{
  "title": "string",
  "introduction": "string",
  "theoretical_foundations": [
    {"heading": "string", "content": "string"}
  ],
  "key_findings_and_methodologies": "string",
  "summary_table_markdown": "string",  // GitHub table with headers:
                                        // | Research Domain | Key Finding | Supporting Studies | Theoretical Implications |
  "conclusion_and_future_directions": "string",
  "references": [
    {
      "category": "one of: Core Theory | Methods/Assessment | Environmental Complexity | Landmarks | Individual Differences | VR vs Real | Neuro/Imaging | Applied/Wayfinding | Developmental",
      "full_citation": "APA-style string (authors, year, title, outlet, vol(issue), pages, DOI/URL if available)",
      "year": 2012,
      "doi": "string or null",
      "url": "string or null"
    }
  ],
  "reference_summary": {
    "total": 0,
    "since_2015": 0,
    "foundational_classics": 0
  }
}

Hard requirements:
- Length: ≥2,800 words across Introduction, Foundations, Key Findings, and Conclusion.
- References: 24–30 total; ≥8 since 2015; include Tolman (1948), Lynch (1960), Siegel & White (1975). Provide DOI if real; else URL. Do NOT invent identifiers.
- Table: ≥12 rows.

Topic-specific: align with the user's prompt (e.g., “Free Exploration → Sketch-Map Recall”), covering classic & modern findings (VR, fMRI/EEG, applied wayfinding).
Return ONLY the JSON object.
"""
    # Escape braces so LangChain doesn't treat schema as template vars
    escaped_system_prompt = system_prompt.replace("{", "{{").replace("}", "}}")

    llm = get_llm(
        model_name=GEMINI_PRO_MODEL_NAME,
        temperature=0.1,
        max_output_tokens=8192
    )

    agent = create_tool_calling_agent(
        llm,
        tools,
        ChatPromptTemplate.from_messages(
            [
                ("system", escaped_system_prompt),
                ("human", "TOPIC:\n{input}\n\nSEED_SOURCES (JSON list; use heavily):\n{seed_sources}"),
                ("placeholder", "{agent_scratchpad}"),
            ]
        ),
    )

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False,                 # quieter console
        handle_parsing_errors=True,
        return_intermediate_steps=False,
        max_iterations=40,
    )

    logger.info("'Gold Standard' Professional Researcher AGENT created successfully.")
    return agent_executor
