import time
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

    def _parse_anthropic_response(self, response):
        """
        Parses the response from the Anthropic model.
        """

        """
        Analisa a resposta do modelo Anthropic, separando o conteúdo em 'thinking' e 'text'.
        
        Args:
            response: A resposta do modelo Anthropic (AIMessageChunk)
            
        Returns:
            Um dicionário com as chaves 'thinking' e 'text', se disponíveis
        """
        result = {"thinking": None, "text": None}
        
        # Verifica se a resposta tem conteúdo
        if not hasattr(response, "content") or not response.content:
            return result
            
        # Verifica se o conteúdo é uma lista
        if isinstance(response.content, list) and len(response.content) > 0:
            for chunk in response.content:
                if isinstance(chunk, dict) and "type" in chunk:
                    # Extrai o pensamento (thinking)
                    if chunk["type"] == "thinking" and "thinking" in chunk:
                        result["thinking"] = chunk["thinking"]
                    
                    # Extrai o texto da resposta
                    elif chunk["type"] == "text" and "text" in chunk:
                        result["text"] = chunk["text"]
        
        # Se o conteúdo for uma string, considera como texto
        elif isinstance(response.content, str):
            result["text"] = response.content
            
        return result
        
    def stream_response(self, prompt, graph, memory_config):
        """
        Faz streaming da resposta do assistente, exibindo pensamentos em tempo real
        e substituindo-os por um expander quando a resposta final começa.
        
        Args:
            prompt: A pergunta do usuário
            graph: O grafo de processamento
            memory_config: Configuração de memória
            
        Returns:
            A resposta final gerada
        """
        final_response = ""
        streaming_thoughts = ""
        thinking_expander_created = False
        at_start = False
        current_node = ""
        
        # Placeholders para atualização em tempo real
        final_placeholder = st.empty()
        thinking_placeholder = st.empty()
        node_placeholder = st.empty()
        loading_placeholder = st.empty()
        
        # Iniciar streaming da resposta
        for response in graph.stream(
            {"messages": [HumanMessage(content=prompt)]},
            stream_mode="messages",
            config=memory_config
        ):
            # Processar a resposta
            if isinstance(response, tuple):
                ai_msg_chunk, metadata = response
                
                # Verificar o nó atual do grafo
                if 'langgraph_node' in metadata:
                    actual_node = metadata['langgraph_node']
                    
                    # Atualizar placeholders com base no nó atual
                    if actual_node == '__start__':
                        if not at_start:
                            at_start = True
                            current_node = actual_node
                            loading_placeholder.markdown("**Entendendo sua pergunta...**")
                            node_placeholder.empty()
                    elif actual_node == 'retrieve_documents':
                        loading_placeholder.empty()
                        node_placeholder.markdown("🔍 **Recuperando documentos...**")
                    elif actual_node == 'grade_documents':
                        loading_placeholder.empty()
                        node_placeholder.markdown("**Avaliando a relevância dos documentos recuperados...**")
                    elif actual_node == 'generate':
                        loading_placeholder.empty()
                        node_placeholder.markdown("🤖 **Gerando resposta...**")
                
                # Processar o conteúdo da mensagem
                if isinstance(ai_msg_chunk, AIMessageChunk) and hasattr(ai_msg_chunk, 'content'):
                    parsed_response = self._parse_anthropic_response(ai_msg_chunk)
                    
                    # Processar pensamentos (thinking)
                    if parsed_response["thinking"] and not thinking_expander_created:
                        streaming_thoughts += parsed_response["thinking"]
                        thinking_placeholder.markdown(
                            f"**O modelo está pensando...**\n\n{streaming_thoughts}"
                        )
                    
                    # Processar texto da resposta
                    if parsed_response["text"]:
                        if not thinking_expander_created:
                            thinking_placeholder.empty()
                            st.session_state.thoughts = streaming_thoughts
                            st.expander("🤖 Pensamentos do Modelo", expanded=False).markdown(
                                st.session_state.thoughts
                            )
                            thinking_expander_created = True
                            node_placeholder.empty()
                        
                        final_response += parsed_response["text"]
                        final_placeholder.markdown(final_response)
            
            # Pequena pausa para melhorar a experiência de streaming
            time.sleep(0.1)
        
        # Limpar placeholders ao terminar
        loading_placeholder.empty()
        node_placeholder.empty()
        
        return final_response

if __name__ == "__main__":
    print(os.environ.get("API_URL"))