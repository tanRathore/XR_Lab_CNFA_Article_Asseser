### START OF FINAL CORRECTED FILE: src/cna_rag_agent/graph_store/query.py ###

import os
import sys
from pathlib import Path

# --- Setup Project Path ---
SRC_DIR = Path(__file__).resolve().parent.parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
# --- End Setup Project Path ---

from langchain_core.prompts import ChatPromptTemplate
# <<< FIX: Import the necessary, stable LCEL components >>>
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_neo4j import Neo4jGraph
from langchain_google_genai import ChatGoogleGenerativeAI

from cna_rag_agent.utils.logging_config import logger
from cna_rag_agent.config import GEMINI_PRO_MODEL_NAME

# --- Paste your Neo4j Credentials Here ---
NEO4J_URI = "neo4j+s://f79d7662.databases.neo4j.io"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "vU9_LsfDXixW3jm0SF-hZZ516-Zkbb_2qsbKYcPXFtE"
# -----------------------------------------

# Prompt to convert a question into a Cypher query.
CYPHER_GENERATION_TEMPLATE = """
You are an expert Neo4j developer. Given a graph schema and a user question, create a Cypher query to answer the question.
Do not return any explanation or apology, just the Cypher query itself.

Schema:
{schema}

Question:
{question}

Cypher Query:
"""
cypher_prompt = ChatPromptTemplate.from_template(CYPHER_GENERATION_TEMPLATE)

# Prompt to synthesize a final answer from the database results.
ANSWER_GENERATION_TEMPLATE = """
You are an expert research assistant. You are given a question and the results from a database query.
Synthesize a concise, natural language answer based on the provided information.

Question:
{question}

Database Results:
{context}

Answer:
"""
answer_prompt = ChatPromptTemplate.from_template(ANSWER_GENERATION_TEMPLATE)

# <<< FIX: The entire function is rewritten with a robust and correct LCEL structure >>>
def create_graph_query_chain():
    """
    Creates a chain that can query the Neo4j graph based on a natural language question
    by manually constructing the chain from fundamental components.
    """
    logger.info("Connecting to Neo4j graph...")
    graph = Neo4jGraph(
        url=NEO4J_URI, 
        username=NEO4J_USERNAME, 
        password=NEO4J_PASSWORD
    )
    graph.refresh_schema()
    logger.info("Neo4j graph schema refreshed.")
    
    logger.info("Creating manual Graph Cypher Chain...")
    llm = ChatGoogleGenerativeAI(model=GEMINI_PRO_MODEL_NAME, temperature=0.0)
    
    # 1. This sub-chain generates the Cypher query.
    #    It expects a dictionary with "schema" and "question" keys.
    generate_cypher_query = (
        cypher_prompt
        | llm
        | StrOutputParser()
    )
    
    # 2. This sub-chain generates the final natural language answer.
    #    It expects a dictionary with "context" and "question" keys.
    generate_answer = (
        answer_prompt
        | llm
        | StrOutputParser()
    )

    # 3. This is the full, orchestrating chain. It is built to correctly
    #    handle a single string input and manage the data flow.
    full_chain = (
        # The input to the chain is the user's question (a string).
        # The first step is to create a dictionary that will be passed along.
        # RunnableParallel (the dict syntax) does this perfectly.
        {
            "question": RunnablePassthrough(), # Pass the original question through
            "schema": lambda x: graph.schema # Get the graph schema
        }
        # The output of this step is: {"question": "...", "schema": "..."}
        | RunnablePassthrough.assign(
            # Generate the Cypher query and add it to the dictionary.
            cypher=generate_cypher_query
          )
        # The output of this step is: {"question": "...", "schema": "...", "cypher": "..."}
        | RunnablePassthrough.assign(
            # Execute the Cypher query to get the database results.
            context=lambda x: graph.query(x["cypher"])
          )
        # The output of this step is: {"question": "...", "schema": "...", "cypher": "...", "context": [...]}
        | generate_answer # Pass the final dictionary to the answer generation chain.
    )

    logger.info("Manual Graph Cypher Chain created successfully.")
    return full_chain

if __name__ == '__main__':
    graph_query_chain = create_graph_query_chain()
    
    question = "List the different experimental paradigms you know about."
    
    print(f"\n--- Testing Graph Query Tool ---")
    print(f">>> Question: {question}")
    
    # The input to the chain is the question string.
    response = graph_query_chain.invoke(question)
    
    print("\n--- Final Answer ---")
    print(response)

### END OF FINAL CORRECTED FILE ###