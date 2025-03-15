from langchain_core.prompts import ChatPromptTemplate

# --- Query Routing ---
ROUTING_SYSTEM_PROMPT = """You are an expert at routing a user question to a vectorstore or web search.
The vectorstore contains documents related to agents, prompt engineering, and adversarial attacks.
Use the vectorstore for questions on these topics. Otherwise, use web-search."""
ROUTING_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", ROUTING_SYSTEM_PROMPT),
        ("human", "{question}"),
    ]
)

# --- Document Grading ---
GRADING_SYSTEM_PROMPT = """You are a grader assessing relevance of a retrieved document to a user question. \n 
    If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
    It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
GRADING_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", GRADING_SYSTEM_PROMPT),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ]
)

# --- Hallucination Grading ---
HALLUCINATION_SYSTEM_PROMPT = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n 
     Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."""
HALLUCINATION_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", HALLUCINATION_SYSTEM_PROMPT),
        ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
    ]
)

# --- Answer Grading ---
ANSWER_SYSTEM_PROMPT = """You are a grader assessing whether an answer addresses / resolves a question \n 
     Give a binary score 'yes' or 'no'. Yes' means that the answer resolves the question."""
ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", ANSWER_SYSTEM_PROMPT),
        ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
    ]
)

# --- Question Rewriting ---
REWRITE_SYSTEM_PROMPT = """You a question re-writer that converts an input question to a better version that is optimized \n 
     for vectorstore retrieval. Look at the input and try to reason about the underlying semantic intent / meaning."""
REWRITE_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", REWRITE_SYSTEM_PROMPT),
        (
            "human",
            "Here is the initial question: \n\n {question} \n Formulate an improved question.",
        ),
    ]
)

# --- RAG ---
RAG_SYSTEM_PROMPT = """You are an expert at elaborating answers using retrieved documents from a vectorstore. \n
    You are given a user question and a set of documents. Answer the question using the documents."""
RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", RAG_SYSTEM_PROMPT),
        ("human", "User question: \n\n {question} \n\n Documents: \n\n {documents}"),
    ]
)