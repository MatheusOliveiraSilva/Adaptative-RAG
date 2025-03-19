import streamlit as st

from langchain_core.messages import HumanMessage, AIMessageChunk, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

import os
from dotenv import load_dotenv
from pathlib import Path

root_dir = Path().absolute()
load_dotenv(dotenv_path=root_dir / ".env")

class AnthropicStreaming:
    """
    Utility class to stream the response from the Anthropic model.
    """
    def __init__(self,
                 model_name: str,
                 thinking_mode: bool = False
                ) -> None:
        
        self.model_name = model_name
        self.thinking_mode = thinking_mode

    def stream_response(self, user_input, graph, memory_config):
        """
        Give user's prompt to langgraph agent and stream the response.
        """

        prompt = {"messages": [HumanMessage(content=user_input)]}

        final_response = ""
        streaming_thoughts = ""
        thinking_expander_created = False
        current_node = ""
        at_start = False
        is_routing = False
        document_relevance_low = False

        st.session_state.thoughts = ""  # Initialize the thoughts state.

        final_placeholder = st.empty()
        thinking_placeholder = st.empty()
        node_placeholder = st.empty()
        loading_placeholder = st.empty()

        for response in graph.stream(
            prompt,
            stream_mode="messages",
            config=memory_config
        ):
            # Exemplos de chunks chegando:
            # (AIMessageChunk(content='', additional_kwargs={}, response_metadata={}, id='run-a731c58b-a29a-4ad3-b7c4-f58253bcc35b'), {'thread_id': '123', 'langgraph_step': 2, 'langgraph_node': 'grade_documents', 'langgraph_triggers': ['retrieve_documents'], 'langgraph_path': ('__pregel_pull', 'grade_documents'), 'langgraph_checkpoint_ns': 'grade_documents:3d421f69-5d5b-4473-f209-f37ae6339c0c', 'checkpoint_ns': 'grade_documents:3d421f69-5d5b-4473-f209-f37ae6339c0c', 'ls_provider': 'openai', 'ls_model_name': 'gpt-4o-mini', 'ls_model_type': 'chat', 'ls_temperature': 0.0})
            
            # Vai ser sempre uma tupla com dois elementos: AIMessageChunk e um dicionário com metadados.
            ai_msg_chunk, metadata = response

            # ===== Processing retrieval chunks from callbacks. =====
            if isinstance(ai_msg_chunk, AIMessageChunk):
                if hasattr(ai_msg_chunk, 'additional_kwargs') and ai_msg_chunk.additional_kwargs:
                    additional_data = ai_msg_chunk.additional_kwargs
                    
                    if 'status' in additional_data:
                        if additional_data['status'] == 'retrieving_started':
                            loading_placeholder.markdown("🔎 **Searching for relevant documents...**")
                
                if ai_msg_chunk.content:
                    final_response += ai_msg_chunk.content
                    final_placeholder.markdown(final_response)

            # ===== End of processing retrieval chunks from callbacks. =====

            if 'langgraph_node' in metadata:
                actual_node = metadata['langgraph_node']

                if actual_node == '__start__':
                    if not at_start:
                        at_start = True
                        current_node = actual_node
                        loading_placeholder.markdown("**Understanding your question...**")
                        node_placeholder.empty()
                    
                elif actual_node != current_node:
                    current_node = actual_node
                    
                    if 'grade_documents' in actual_node:
                        loading_placeholder.empty()
                        node_placeholder.markdown("🔍 **Avaliando relevância dos documentos recuperados...**")
                    elif document_relevance_low and 'retrieve_documents' in actual_node:
                        loading_placeholder.empty()
                        node_placeholder.markdown("📚 **Relevância dos documentos é baixa, buscando mais documentos...**")
                    elif 'generate' in actual_node:
                        loading_placeholder.empty()
                        node_placeholder.markdown("🤖 **Gerando resposta...**")
                
                # Verificar se os documentos têm baixa relevância
                if 'langgraph_node' in metadata and 'grade_documents' in metadata['langgraph_node']:
                    if hasattr(ai_msg_chunk, 'additional_kwargs') and 'parsed' in ai_msg_chunk.additional_kwargs:
                        parsed_data = ai_msg_chunk.additional_kwargs['parsed']
                        if hasattr(parsed_data, 'binary_score') and parsed_data.binary_score == 'no':
                            document_relevance_low = True

        # Limpar placeholders ao terminar
        loading_placeholder.empty()
        node_placeholder.empty()
                    
        return final_response


    def _parse_anthropic_response(self, response):
        """
        Parses the response from the Anthropic model.
        """
        pass

if __name__ == "__main__":
    print(os.environ.get("API_URL"))