from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from pinecone import Pinecone
from pathlib import Path
from langchain_core.documents import Document
from langchain_community.tools.tavily_search import TavilySearchResults
from agent.langgraph.states import GraphState
from agent.langgraph.chains import (
    question_rewriter_chain, answer_grader_chain, 
    document_grader_chain, hallucination_grader_chain, 
    query_router_chain, rag_chain
)

root_dir = Path().absolute()

load_dotenv(dotenv_path=root_dir / ".env")

class AdaptiveRAGNodes:
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

    def retrieve_documents(self, state: GraphState) -> GraphState:
        """
        Retrieve documents from the vectorstore.
        
        Args:
            state: GraphState with current state (only question user question).

        Returns:
            GraphState with documents and question
        """
        print(f"--- Retrieving documents ---")
        docs = self.retriever.invoke(state["question"])

        return {"documents": docs, "question": state["question"]}
    
    def format_docs(self, docs: list[Document]) -> str:
        return "\n\n".join(doc.page_content for doc in docs)
    
    def generate(self, state: GraphState) -> GraphState:
        """
        Generate an answer to the user question, using the documents and the question.
        
        Args:
            state: GraphState with current state (documents and question).

        Returns:
            GraphState with answer
        """
        print(f"--- Generating answer ---")
        generation = rag_chain.invoke({"question": state["question"], "documents": self.format_docs(state["documents"])})
        return {"documents": state["documents"], "question": state["question"], "generation": generation}

    def grade_documents(self, state: GraphState) -> GraphState:
        """
        Grade the documents based on the user question.
        
        Args:
            state: GraphState with current state (documents and question).

        Returns:
            GraphState with documents and question
        """

        print(f"--- Grading documents ---")
        filtered_docs = []
        for doc in state["documents"]:
            isRelevant = document_grader_chain.invoke({"document": self.format_docs([doc]), "question": state["question"]})
            if isRelevant.binary_score == "yes":
                filtered_docs.append(doc)

        return {"documents": filtered_docs, "question": state["question"]}

    def rewrite_query(self, state: GraphState) -> GraphState:
        """
        Rewrite the user question to be more specific and to the point.
        
        Args:
            state: GraphState with current state (question).

        Returns:
            GraphState with better question
        """
        print(f"--- Rewriting query ---")
        better_query = question_rewriter_chain.invoke({"question": state["question"]})
        return {"question": better_query, "documents": state["documents"]}
    
    def web_search(self, state: GraphState) -> GraphState:
        """
        Search the web for the user question.
        
        Args:
            state: GraphState with current state (question).
        """
        print(f"--- Searching web ---")
        docs = self.web_search_tool.invoke({"query": state["question"]})

        web_results = "\n".join([d["content"] for d in docs])
        web_results_doc = Document(page_content=web_results, metadata={"source": "web"})

        return {"documents": [web_results_doc], "question": state["question"]}
    
    def grade_generation(self, state: GraphState) -> GraphState:
        """
        Grade the generation based on the user question.
        
        Args:
            state: GraphState with current state (question).
        """
        print(f"--- Grading generation ---")
        is_grounded = hallucination_grader_chain.invoke(
            {
                "documents": self.format_docs(state["documents"]), 
                "generation": state["generation"]
            }
        )

        if is_grounded.binary_score == "yes":
            isUseful = answer_grader_chain.invoke(
                {
                    "question": state["question"],
                    "generation": state["generation"]
                }
            )

            if isUseful.binary_score == "yes":
                return "useful"
            else:
                return "not_useful"
            
        else:
            return "not supported"
        
if __name__ == "__main__":
    nodes = AdaptiveRAGNodes()
    print(nodes.retrieve_documents("AI Engineering"))