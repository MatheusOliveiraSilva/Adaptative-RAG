from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from agent.lang_graph.output_models import RouteQuery, GradeDocuments, GradeHallucinations, GradeAnswer
from agent.lang_graph.prompts import (
    ROUTING_PROMPT, GRADING_PROMPT, HALLUCINATION_PROMPT, 
    ANSWER_PROMPT, REWRITE_PROMPT, RAG_PROMPT
)
from dotenv import load_dotenv
from pathlib import Path
import os

root_dir = Path().absolute()

load_dotenv(dotenv_path=root_dir / ".env")

# --- LLM Instances ---
# Use the OPENAI_API_KEY environment variable explicitly
GPT_4O_MINI = ChatOpenAI(
    model="gpt-4o-mini", 
    temperature=0,
    openai_api_key=os.environ.get("OPENAI_API_KEY")
)

# --- Chains ---
query_router_llm = GPT_4O_MINI.with_structured_output(RouteQuery)
query_router_chain = ROUTING_PROMPT | query_router_llm

document_grader_llm = GPT_4O_MINI.with_structured_output(GradeDocuments)
document_grader_chain = GRADING_PROMPT | document_grader_llm

hallucination_grader_llm = GPT_4O_MINI.with_structured_output(GradeHallucinations)
hallucination_grader_chain = HALLUCINATION_PROMPT | hallucination_grader_llm

answer_grader_llm = GPT_4O_MINI.with_structured_output(GradeAnswer)
answer_grader_chain = ANSWER_PROMPT | answer_grader_llm

question_rewriter_chain = REWRITE_PROMPT | GPT_4O_MINI | StrOutputParser()

rag_chain = RAG_PROMPT | GPT_4O_MINI | StrOutputParser()