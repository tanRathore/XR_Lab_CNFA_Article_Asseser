# -*- coding: utf-8 -*-
# FINAL: src/cna_rag_agent/agent/orchestrator.py
# Adds a "/report ..." slash command that FORCE-routes to the lead_researcher_agent
# and injects an input that asks for JSON-only long-form output.

import sys
from pathlib import Path
from operator import itemgetter

# --- Setup Project Path ---
SRC_DIR = Path(__file__).resolve().parent.parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
# --- End Setup Project Path ---

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableBranch, RunnableLambda
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

from cna_rag_agent.utils.logging_config import logger
from cna_rag_agent.generation.generator import get_llm
from cna_rag_agent.config import GEMINI_PRO_MODEL_NAME

# Specialists (ensure these names are exported from specialists.py)
from cna_rag_agent.agent.specialists import (
    information_agent,
    list_articles_agent,
    deep_dive_agent,
    open_questions_agent,
    lead_researcher_agent,   # long-form report generator (JSON structured)
    data_scientist_agent,
    visualizer_agent,
    comparative_analyst_agent,
    online_researcher_agent,
    general_qa_agent
)

# --- simple in-memory store for chat histories (keyed by session_id) ---
_SESSION_STORE = {}

def _get_session_history(session_id: str) -> ChatMessageHistory:
    """Return a ChatMessageHistory bucket for this session_id."""
    if session_id not in _SESSION_STORE:
        _SESSION_STORE[session_id] = ChatMessageHistory()
    return _SESSION_STORE[session_id]


def create_advanced_agent_router():
    """
    Chief-of-Staff router with:
      - /report <topic> slash-command that force-routes to lead_researcher_agent
      - normal classifier-based routing otherwise
      - full conversational memory via RunnableWithMessageHistory
    """
    logger.info("Creating the final advanced agent router with all specialists...")

    # -------- Classifier with chat history context --------
    classification_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a precise router. Read the ongoing conversation and the latest user question, "
         "then choose the SINGLE most specific tool category."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human",
         """Given the user's question, classify it into one of:
- 'information': general overview/summary/intro
- 'list_articles': list or find articles FROM THE LOCAL DOCS
- 'deep_dive': detailed methods / experimental setups
- 'open_questions': future research or gaps based on local docs
- 'lead_researcher_query': deep, structured queries or requests for long-form literature reviews / reports
- 'data_science': calculations / data analysis
- 'visualization': create an image or diagram
- 'comparison': compare and contrast two or more topics
- 'online_research': explicitly asks to search the web or find recent items
- 'general_qa': simple definitions or quick facts from local docs

Return a JSON object with a single key 'tool' and the chosen category as the value.

Question: {question}
JSON Response:""")
    ])

    llm = get_llm(model_name=GEMINI_PRO_MODEL_NAME, temperature=0.0)
    classification_chain = classification_prompt | llm | JsonOutputParser() | itemgetter("tool")

    # -------- Normal router (used when NOT slash-forced) --------
    normal_router = RunnableBranch(
        (lambda x: x.get("tool") == "information",            information_agent),
        (lambda x: x.get("tool") == "list_articles",          list_articles_agent),
        (lambda x: x.get("tool") == "deep_dive",              deep_dive_agent),
        (lambda x: x.get("tool") == "open_questions",         open_questions_agent),
        (lambda x: x.get("tool") == "lead_researcher_query",  lead_researcher_agent),
        (lambda x: x.get("tool") == "data_science",           data_scientist_agent),
        (lambda x: x.get("tool") == "visualization",          visualizer_agent),
        (lambda x: x.get("tool") == "comparison",             comparative_analyst_agent),
        (lambda x: x.get("tool") == "online_research",        online_researcher_agent),
        general_qa_agent  # default / fallback
    )

    normal_chain = {
        "tool":          classification_chain,
        "input":         itemgetter("question"),
        "chat_history":  itemgetter("chat_history")
    } | normal_router

    # -------- Slash command preprocess --------
    def _slash_preprocess(payload: dict):
        """
        Detects '/report ...' and prepares a forced JSON-spec prompt for the lead researcher.
        Returns a dict with:
          - slash_report: bool
          - prepared_input: str (if slash_report == True)
          - question, chat_history: passthrough for non-slash
        """
        q = (payload.get("question") or "").strip()
        hist = payload.get("chat_history")

        # Supports:
        #   /report <topic>
        #   /report: <topic>
        #   /report   <topic>   (extra spaces)
        is_slash = q.lower().startswith("/report")
        if is_slash:
            # Extract topic after the '/report' token (tolerate ':')
            rest = q[len("/report"):].lstrip()
            if rest.startswith(":"):
                rest = rest[1:].lstrip()
            topic = rest or "Topic unspecified"

            # Very explicit, topic-agnostic JSON-only instruction to guarantee exportable payload.
            prepared = f"""
Write a comprehensive literature review grounded in a large, relevant bibliography on the following topic.

TOPIC:
{topic}

Return ONLY valid JSON with EXACT schema:

{{
  "title": "string",
  "introduction": "multi-paragraph text (≥350 words)",
  "theoretical_foundations": [
    {{"heading": "string", "content": "multi-paragraph text (≥250 words)"}},
    {{"heading": "string", "content": "multi-paragraph text (≥250 words)"}},
    {{"heading": "string", "content": "multi-paragraph text (≥250 words)"}}
  ],
  "key_findings_and_methodologies": "multi-paragraph text (≥700 words)",
  "summary_table_markdown": "| Research Domain | Key Finding | Supporting Studies | Theoretical Implications |\\n|---|---|---|---|\\n| ... | ... | ... | ... |",
  "conclusion_and_future_directions": "multi-paragraph text (≥300 words) with explicit testable proposals",
  "references": [
    {{
      "category": "freeform topical label (e.g., Core Theory, Methods, Constructs, Mechanisms, Applications, Debates, Reviews/Meta, etc.)",
      "full_citation": "APA-style citation",
      "year": 2019,
      "doi": "10.xxxx/xxxxx or null",
      "url": "https://... or null"
    }}
  ]
}}

Quality constraints:
- Length target: ≈2,800–3,400 words overall.
- Table: 10–14 substantive rows.
- Relevance over publication year; include classics + contemporary.
- NO prose outside the JSON object.
"""
            return {
                "slash_report": True,
                "prepared_input": prepared,
                "chat_history": hist
            }

        # Not a slash report -> just pass through
        return {
            "slash_report": False,
            "question": q,
            "chat_history": hist
        }

    preprocess = RunnableLambda(_slash_preprocess)

    # -------- Forced lead-researcher path for /report --------
    forced_lead_chain = {
        "input":        itemgetter("prepared_input"),
        "chat_history": itemgetter("chat_history")
    } | lead_researcher_agent

    # -------- Combine: if slash_report -> forced lead; else -> normal classify+route --------
    combined_core = preprocess | RunnableBranch(
        (lambda x: x.get("slash_report") is True, forced_lead_chain),
        normal_chain
    )

    # -------- Wrap with memory --------
    router_with_history = RunnableWithMessageHistory(
        combined_core,
        _get_session_history,
        input_messages_key="question",     # used when normal_chain runs
        history_messages_key="chat_history",
        output_messages_key="output"
    )

    logger.info("Final advanced agent router created successfully (slash support enabled).")
    return router_with_history
