
from langchain_core.prompts import PromptTemplate

# Basic QA Prompt - instructs the LLM to answer based only on context.
QA_TEMPLATE_STR = """You are an expert AI assistant for answering questions based on provided technical documents.
Your goal is to provide accurate and concise answers derived *only* from the context below.
If the information to answer the question is not in the context, state clearly: "I cannot answer this question based on the provided documents."
Do not make up information or answer from outside the given context.

Context:
{context}

Question: {question}

Answer:
"""

QA_PROMPT = PromptTemplate.from_template(QA_TEMPLATE_STR)


# Add more prompt templates here as needed, for example:
# - For the two-step article selection
# - For generating the 3-page summaries
# - For extracting "stimuli duplication" information

THREE_PAGER_SUMMARY_TEMPLATE_STR = """You are a scientific assistant tasked with creating a detailed "3-pager" summary of the following technical article content.
The summary should enable another scientist to quickly grasp the article's core concepts and understand how to reproduce its key experiments/stimuli if applicable.

Article Content:
{article_text}

Based *only* on the provided article content, generate a comprehensive summary with the following sections:
1.  **Introduction & Problem Statement:** Briefly explain the context and the problem the article addresses.
2.  **Key Methodologies:** Describe the main methods, approaches, and techniques used. If mathematical formalisms are central, briefly describe their purpose.
3.  **Core Findings & Results:** Summarize the most important outcomes, discoveries, and data presented.
4.  **Detailed Experimental Setup / Stimuli Duplication (if applicable):**
    * If the article describes experiments or stimuli, extract all relevant details needed for duplication.
    * This includes: materials, equipment (and model numbers/versions if provided), reagents, software (and versions if provided), hardware configurations, specific parameters (e.g., concentrations, temperatures, durations, frequencies, visual angles, luminance levels, participant instructions, dataset details and accessibility if mentioned).
    * If procedures are detailed, summarize them.
    * If this information is not present or not applicable, state "Experimental/stimuli duplication details are not provided or not applicable."
5.  **Discussion & Implications:** Briefly discuss the significance of the findings as stated by the authors, limitations mentioned, and potential future work suggested.
6.  **Key Figures/Tables to Reference (Optional but helpful):** If specific figures or tables are central to understanding the main points, list them (e.g., "Figure 1 shows...", "Table 2 summarizes...").

The overall summary should be detailed yet concise, aiming for the conceptual equivalent of 2-3 pages of text.
Focus on extracting factual information relevant to scientific understanding and potential reproduction. Do not add external knowledge.
If parts of the article content are unclear or seem incomplete for a section, note that in your summary for that section.

Begin Summary:
"""

THREE_PAGER_SUMMARY_PROMPT = PromptTemplate.from_template(THREE_PAGER_SUMMARY_TEMPLATE_STR)

# You can also add a specific prompt for the first step of article selection, e.g.,
# "Given the user query '{user_query}' and the following article titles and abstracts: {articles_info},
# which articles are most relevant to answer the query? List their identifiers."