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

# Explicitly use the OpenAI API key
summary_llm = ChatOpenAI(
    model="gpt-4o-mini-2024-07-18",
    openai_api_key=os.environ.get("OPENAI_API_KEY")
)

def stream_assistant_response(prompt, graph, memory_config) -> str:
    """
    Stream assistant answer displaying thoughts in real time and, when the final
    answer starts, replacing thoughts with an expander. Returns the final generated
    answer.
    """
    final_response = ""
    streaming_thoughts = ""
    thinking_expander_created = False

    # Reinicia os pensamentos para a interação atual (não acumula com interações anteriores)
    st.session_state.thoughts = ""

    # Placeholders para atualização em tempo real
    final_placeholder = st.empty()
    thinking_placeholder = st.empty()

    for response in graph.stream(
        {"question": prompt},
        stream_mode="messages",
        config=memory_config
    ):
        if isinstance(response, tuple):
            for item in response:
                if isinstance(item, AIMessageChunk) and item.content:

                    chunk = item.content[0]
                    if "type" in chunk:
                        if chunk["type"] == "thinking" and "thinking" in chunk:
                            if not thinking_expander_created:
                                streaming_thoughts += chunk["thinking"]
                                thinking_placeholder.markdown(
                                    f"**Model is thinking...**\n\n{streaming_thoughts}"
                                )
                        elif chunk["type"] == "text" and "text" in chunk:
                            if not thinking_expander_created:
                                thinking_placeholder.empty()
                                st.session_state.thoughts = streaming_thoughts
                                st.expander("🤖 Model's Thoughts", expanded=False).markdown(
                                    st.session_state.thoughts
                                )
                                thinking_expander_created = True
                            final_response += chunk["text"]
                            final_placeholder.markdown(final_response)
        time.sleep(0.3)

    return final_response

def convert_messages_to_save(messages: list) -> list:
    """
    Convert the messages list to a list of lists with the following structure:
      1) 'user' (HumanMessage)
      2) 'assistant_thought' (AIMessage)
      3) 'assistant_response' (AIMessage)

    We can only do this because we know the order of the messages in the list. It's the anthropic format when we allow the thinking mode, wich is the case.
    """
    messages_to_save = []
    i = 0
    n = len(messages)

    while i < n:
        if i % 3 == 0:
            messages_to_save.append(["user", messages[i].content])
            i += 1
        elif i % 3 == 1:
            messages_to_save.append(["assistant_thought", messages[i].content])
            i += 1
        elif i % 3 == 2:
            messages_to_save.append(["assistant_response", messages[i].content])
            i += 1

    return messages_to_save

def summary_conversation_theme(prompt: str) -> str:
    """
    Summarize the conversation theme based on the user's first message.
    """

    summary_prompt = "Take the user input prompt and resume it in a few words as the main theme of the conversation. Try to use less min2 max5 words. User prompt: {prompt}"

    chain = summary_llm | StrOutputParser()

    theme = chain.invoke(summary_prompt.format(prompt=prompt))

    return theme if theme else "General Chat"

if __name__ == "__main__":
    print(summary_conversation_theme("Talk about HyDE"))