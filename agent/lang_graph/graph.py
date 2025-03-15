from langgraph.graph import END, StateGraph, START
from langgraph.checkpoint.memory import MemorySaver
from agent.lang_graph.states import GraphState
from agent.lang_graph.nodes import AdaptiveRAGNodes
from agent.lang_graph.edges import AdaptiveRAGEdges

class AdaptiveRAGGraph:
    def __init__(self):
        self.nodes = AdaptiveRAGNodes()
        self.edges = AdaptiveRAGEdges()

        self.Graph = StateGraph(GraphState)

        self.Graph = self.setup_nodes(self.Graph)
        self.Graph = self.setup_edges(self.Graph)

        self.agent = self.Graph.compile(checkpointer=MemorySaver())

    def setup_nodes(self, graph: StateGraph) -> StateGraph:
        graph.add_node("web_search", self.nodes.web_search)
        graph.add_node("retrieve_documents", self.nodes.retrieve_documents)
        graph.add_node("grade_documents", self.nodes.grade_documents)
        graph.add_node("generate", self.nodes.generate)
        graph.add_node("grade_generation", self.nodes.grade_generation)
        graph.add_node("rewrite_query", self.nodes.rewrite_query)

        return graph
    
    def setup_edges(self, graph: StateGraph) -> StateGraph:
        
        graph.add_conditional_edges(
            START,
            self.edges.route_question,
            {
                "web_search": "web_search",
                "vectorstore": "retrieve_documents",
            },
        )

        graph.add_edge("web_search", "generate")
        graph.add_edge("retrieve_documents", "grade_documents")

        graph.add_conditional_edges(
            "grade_documents",
            self.edges.decide_to_generate,
            {
                "rewrite_query": "rewrite_query",
                "generate": "generate",
            },
        )
        graph.add_edge("rewrite_query", "retrieve_documents")
        graph.add_conditional_edges(
            "generate",
            self.nodes.grade_generation,
            {
                "not supported": "generate",
                "useful": END,
                "not useful": "rewrite_query",
            },
        )

        return graph
