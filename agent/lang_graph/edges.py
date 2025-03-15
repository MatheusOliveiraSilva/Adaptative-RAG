from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from pinecone import Pinecone
from pathlib import Path
from langchain_core.documents import Document
from langchain_community.tools.tavily_search import TavilySearchResults
from agent.lang_graph.states import GraphState
from agent.lang_graph.chains import query_router_chain
import os

root_dir = Path().absolute()

load_dotenv(dotenv_path=root_dir / ".env")

class AdaptiveRAGEdges:
    def __init__(self):
        # --- Pinecone Setup ---
        self.pc = Pinecone()
        self.index = self.pc.Index("web-ai-engineer-index")
        # Explicitly use the OpenAI API key
        self.embedding_model = OpenAIEmbeddings(
            model="text-embedding-3-large",
            openai_api_key=os.environ.get("OPENAI_API_KEY")
        )
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
        print(f"--- Routing question ---")
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
        print(f"--- Deciding to generate ---")
        filtered_docs = state["documents"]
        if not filtered_docs:
            return "rewrite_query"
        
        else:
            return "generate"