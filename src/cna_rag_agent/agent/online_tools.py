### START OF FULL FILE: src/cna_rag_agent/agent/online_tools.py ###

import sys
from pathlib import Path
import time

# --- Setup Project Path ---
SRC_DIR = Path(__file__).resolve().parent.parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
# --- End Setup Project Path ---

from langchain.agents import Tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

# This is a placeholder for the real search tool which will be provided by the environment
class GoogleSearchAPI:
    def search(self, queries: list[str]):
        print(f"--- MOCK SEARCH: Pretending to search for: {queries} ---")
        # In a real run, this would return actual search results.
        # We simulate finding a few relevant-looking links.
        return [
            {"url": "https://www.example-university.edu/paper1.pdf", "title": "A Study on Environmental Priming and Creative Cognition", "snippet": "Our research explores how architectural features prime creative thinking..."},
            {"url": "https://www.research-journal.org/review-on-priming", "title": "The Role of Priming in Cognitive Tasks: A Review", "snippet": "This review summarizes the state of research on priming..."},
            {"url": "https://www.neuroscience-today.com/open-questions-in-priming", "title": "Open Questions in Environmental Priming Research", "snippet": "Despite progress, several open questions remain, including the long-term effects of priming..."},
        ]

class BrowsingAPI:
    def browse(self, url: str):
        print(f"--- MOCK BROWSE: Pretending to read content from: {url} ---")
        # In a real run, this would return the text content of the URL.
        # We simulate finding relevant text.
        if "paper1" in url:
            return "This paper details an experiment where participants in high-ceiling rooms showed a 15% increase in divergent thinking scores. The study concludes that spatial volume is a significant factor in priming for creative tasks."
        elif "review" in url:
            return "A major review of the literature indicates that while short-term priming effects are well-documented, the durability of these effects is a significant open question. Future work should focus on longitudinal studies."
        elif "open-questions" in url:
            return "Key open questions in environmental priming include: 1) The role of individual differences in susceptibility to priming. 2) The mechanisms by which abstract primes translate to concrete behaviors. 3) The potential for negative or counter-productive priming in poorly designed spaces."
        return "No relevant content found."

# Instantiate our mock tools
google_search = GoogleSearchAPI()
browsing = BrowsingAPI()


from cna_rag_agent.utils.logging_config import logger
from cna_rag_agent.generation.generator import get_llm
from cna_rag_agent.config import GEMINI_PRO_MODEL_NAME

def search_for_articles_and_summarize(topic: str) -> str:
    """
    Performs a comprehensive online search for a given research topic.
    It formulates multiple search queries, browses the top results, and synthesizes
    the findings into a structured report including relevant articles and open questions.
    """
    logger.info(f"Starting comprehensive online search for topic: '{topic}'")
    
    # 1. Formulate diverse search queries
    search_queries = [
        f'"{topic}" scientific review article',
        f'"{topic}" recent research papers',
        f'"{topic}" open research questions',
        f'"{topic}" experimental methods'
    ]
    
    # 2. Execute the searches
    try:
        search_results = google_search.search(queries=search_queries)
        if not search_results:
            return "Could not find any relevant articles online for this topic."
    except Exception as e:
        logger.error(f"Error during web search: {e}", exc_info=True)
        return "An error occurred while trying to search the web."

    # 3. Browse the top results and collect context
    all_context = ""
    urls_to_browse = [res for res in search_results[:3]] # Limit to top 3 for this example
    
    for result in urls_to_browse:
        try:
            content = browsing.browse(url=result["url"])
            all_context += f"--- Source: {result['title']} ({result['url']}) ---\n"
            all_context += content + "\n\n"
            time.sleep(1) # Simulate network delay
        except Exception as e:
            logger.warning(f"Could not browse URL {result['url']}: {e}")

    if not all_context:
        return "Found some search results, but was unable to read their content."

    # 4. Synthesize the final report using an LLM
    synthesis_prompt_template = """
    You are an expert research analyst. Based on the provided context from multiple web search results,
    your task is to synthesize a single, comprehensive report for the user.
    The report must be structured into two distinct sections:
    
    1.  **Relevant Articles Found:** A bulleted list of the most relevant articles, including their titles and a brief, one-sentence summary of their key findings based on the context.
    2.  **Summary of Open Questions:** A paragraph summarizing the key open questions, future research directions, or gaps in the literature that were identified across the sources.

    Do not include any information that is not supported by the provided context.

    Original User Topic:
    {topic}

    Context from Web Search:
    {context}

    Your Synthesized Report:
    """
    synthesis_prompt = ChatPromptTemplate.from_template(synthesis_prompt_template)
    llm = get_llm(model_name=GEMINI_PRO_MODEL_NAME, temperature=0.2)
    synthesis_chain = synthesis_prompt | llm | StrOutputParser()
    
    final_report = synthesis_chain.invoke({
        "topic": topic,
        "context": all_context
    })
    
    logger.info(f"Successfully generated synthesized report for topic: '{topic}'")
    return final_report


# Wrap the function in a LangChain Tool for our agents to use
online_researcher_tool = Tool(
    name="online_article_researcher",
    func=search_for_articles_and_summarize,
    description="""
    Use this tool to find recent articles or identify open research questions on a specific topic by searching the web.
    This is best for when the user's question cannot be answered by the local document library or when they ask for up-to-date information.
    The input should be a concise research topic, for example: 'Environmental Priming and Creative Tasks'.
    """
)


# --- Test Block ---
if __name__ == "__main__":
    logger.info("--- Testing the Online Researcher Tool ---")
    
    test_topic = "Environmental Priming and its Effect on Creative Cognition"
    
    print(f"\n>>> Performing comprehensive search for: '{test_topic}'")
    
    report = online_researcher_tool.invoke(test_topic)
    
    print("\n--- FINAL SYNTHESIZED REPORT ---")
    print(report)

### END OF FULL FILE ###