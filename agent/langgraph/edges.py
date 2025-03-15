from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from pinecone import Pinecone
from pathlib import Path
from langchain_core.documents import Document
from langchain_community.tools.tavily_search import TavilySearchResults
from states import GraphState
from chains import (
    question_rewriter_chain, answer_grader_chain, 
    document_grader_chain, hallucination_grader_chain, 
    query_router_chain, rag_chain
)

root_dir = Path().absolute()

load_dotenv(dotenv_path=root_dir / ".env")

class AdaptiveRAGEdges:
    def __init__(self):
        # --- Pinecone Setup ---
        self.pc = Pinecone()
        self.index = self.pc.Index("web-ai-engineer-index")
        self.embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")
        self.vector_store = PineconeVectorStore(index=self.index, embedding=self.embedding_model)

        # --- Web and Vectorstore Retrievers ---
        self.web_search_tool = TavilySearchResults(k=3)
        self.retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 15}
        )

    def route_question(self, state: GraphState) -> GraphState:
        """
        Route the user question to the appropriate tool.
        
        Args:
            state: GraphState with current state (question).
        """

        route = query_router_chain.invoke({"question": state["question"]})

        if route.datasource == "web_search":
            return "web_search"
        elif route.datasource == "vectorstore":
            return "vectorstore"
        
    def decide_to_generate(self, state: GraphState) -> GraphState:
        """
        Decide whether to generate an answer to the user question.
        
        Args:
            state: GraphState with current state (question).
        """
        
        filtered_docs = state["documents"]
        if not filtered_docs:
            return "rewrite_query"
        
        else:
            return "generate"