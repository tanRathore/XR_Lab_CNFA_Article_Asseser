import os
import sys
from pathlib import Path
from typing import List, Optional

# <<< FIX: Import from pydantic.v1 to resolve the deprecation warning >>>
from pydantic.v1 import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from neo4j import GraphDatabase

from langchain_core.output_parsers.openai_tools import PydanticToolsParser
from langchain_core.runnables import RunnableLambda

# --- Setup Project Path ---
SRC_DIR = Path(__file__).resolve().parent.parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
# --- End Setup Project Path ---

from cna_rag_agent.utils.logging_config import logger
from cna_rag_agent.data_ingestion.loader import load_all_documents
from cna_rag_agent.data_ingestion.preprocessor import chunk_all_document_elements
from cna_rag_agent.config import GEMINI_PRO_MODEL_NAME

# --- Paste your Neo4j Credentials Here ---
NEO4J_URI = "neo4j+s://f79d7662.databases.neo4j.io"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "vU9_LsfDXixW3jm0SF-hZZ516-Zkbb_2qsbKYcPXFtE"
# -----------------------------------------

class Paradigm(BaseModel):
    """A node representing a specific experimental paradigm in CNfA."""
    name: str = Field(description="The unique name of the experimental paradigm, like 'Stressor → Recovery' or 'Exploration → Navigation → Recall'.")
    description: Optional[str] = Field(description="A brief description of what the paradigm studies.")

class Category(BaseModel):
    """A node representing a broad category of research paradigms."""
    name: str = Field(description="The name of the category, e.g., 'Spatial Interaction & Navigation Paradigms'.")

class Method(BaseModel):
    """A node representing a specific method or measurable used in a paradigm."""
    name: str = Field(description="The name of the method or measurable, e.g., 'Heart Rate Variability (HRV)', 'Trier Social Stress Test', 'Sketch Mapping'.")

# <<< FIX: Simplified relationship models to use names (strings) instead of nested objects >>>
class BelongsTo(BaseModel):
    """A relationship indicating that a Paradigm belongs to a Category."""
    paradigm_name: str = Field(description="The name of the source paradigm.")
    category_name: str = Field(description="The name of the target category.")

class UsesMethod(BaseModel):
    """A relationship indicating that a Paradigm uses a specific Method."""
    paradigm_name: str = Field(description="The name of the source paradigm.")
    method_name: str = Field(description="The name of the target method.")

class Graph(BaseModel):
    """A model to hold all extracted nodes and relationships."""
    paradigms: List[Paradigm]
    categories: List[Category]
    methods: List[Method]
    paradigm_to_category_rels: List[BelongsTo]
    paradigm_to_method_rels: List[UsesMethod]

def get_graph_extraction_chain():
    logger.info("Setting up graph extraction chain...")
    llm = ChatGoogleGenerativeAI(model=GEMINI_PRO_MODEL_NAME, temperature=0.0)
    llm_with_graph_tool = llm.bind_tools([Graph], tool_choice="Graph")
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert in Cognitive Neuroscience for Architecture (CNfA). 
        Your task is to identify and extract key entities and their relationships from the provided text, calling the 'Graph' tool to structure them into a knowledge graph.
        The entities are Paradigms, Categories, and Methods.
        The relationships are 'BELONGS_TO' (from Paradigm to Category) and 'USES_METHOD' (from Paradigm to Method).
        When creating relationships, use the exact names of the source and target entities.
        You must provide a value, even if it's an empty list [], for all fields in the Graph tool.
        """),
        ("human", "Extract the graph from the following text:\n\n{chunk}")
    ])
    parser = PydanticToolsParser(tools=[Graph])
    def get_first_graph(graphs: List[Graph]) -> Optional[Graph]:
        return graphs[0] if graphs else None
    chain = prompt | llm_with_graph_tool | parser | RunnableLambda(get_first_graph)
    logger.info("Graph extraction chain setup complete.")
    return chain

def write_graph_to_neo4j(graph: Graph, driver):
    logger.info("Writing extracted graph to Neo4j...")
    if not graph:
        logger.warning("Graph object is None, skipping write to Neo4j.")
        return
    with driver.session() as session:
        if graph.categories:
            for category in graph.categories:
                session.run("MERGE (c:Category {name: $name})", name=category.name)
        if graph.methods:
            for method in graph.methods:
                session.run("MERGE (m:Method {name: $name})", name=method.name)
        if graph.paradigms:
            for paradigm in graph.paradigms:
                session.run("MERGE (p:Paradigm {name: $name, description: $description})", 
                            name=paradigm.name, description=paradigm.description or "")
        
        # <<< FIX: Updated Cypher queries to use the simplified relationship models >>>
        if graph.paradigm_to_category_rels:
            for rel in graph.paradigm_to_category_rels:
                session.run("""
                    MATCH (p:Paradigm {name: $p_name})
                    MATCH (c:Category {name: $c_name})
                    MERGE (p)-[:BELONGS_TO]->(c)
                """, p_name=rel.paradigm_name, c_name=rel.category_name)
        if graph.paradigm_to_method_rels:
            for rel in graph.paradigm_to_method_rels:
                session.run("""
                    MATCH (p:Paradigm {name: $p_name})
                    MATCH (m:Method {name: $m_name})
                    MERGE (p)-[:USES_METHOD]->(m)
                """, p_name=rel.paradigm_name, m_name=rel.method_name)
    logger.info("Successfully wrote graph to Neo4j.")

if __name__ == "__main__":
    logger.info("--- Starting Neo4j Graph Ingestion Pipeline ---")
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
        driver.verify_connectivity()
        logger.info("Successfully connected to Neo4j Aura database.")
    except Exception as e:
        logger.error(f"Failed to connect to Neo4j: {e}", exc_info=True)
        sys.exit(1)
    
    # Clear the database before starting a full ingestion
    logger.info("Clearing existing database...")
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
    logger.info("Database cleared.")

    extraction_chain = get_graph_extraction_chain()
    documents_dir = SRC_DIR.parent / "data" / "raw_documents"
    all_elements = load_all_documents(documents_dir)
    handbook_path_name = "BEST_Handbook of CNfA Experimental Paradigms.docx"
    handbook_elements = [e for e in all_elements if e.metadata.get("file_name") == handbook_path_name]
    if not handbook_elements:
        logger.error(f"CRITICAL: Handbook ('{handbook_path_name}') not found. Cannot perform ingestion.")
        sys.exit(1)
    handbook_chunks = chunk_all_document_elements(handbook_elements)
    
    chunks_to_process = handbook_chunks 
    logger.info(f"Processing {len(chunks_to_process)} chunks from the handbook...")
    
    for i, chunk in enumerate(chunks_to_process):
        logger.info(f"--- Processing Chunk {i+1}/{len(chunks_to_process)} ---")
        try:
            extracted_graph = extraction_chain.invoke({"chunk": chunk.page_content})
            if extracted_graph:
                write_graph_to_neo4j(extracted_graph, driver)
            else:
                logger.warning("No graph data extracted from this chunk.")
        except Exception as e:
            logger.error(f"Error processing chunk {i+1}: {e}", exc_info=True)
            
    logger.info("--- Neo4j Graph Ingestion Pipeline Finished ---")
    driver.close()

### END OF FINAL CORRECTED FILE ###