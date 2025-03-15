import sys
import os

os.system("pip uninstall -y pinecone-plugin-inference || true")

# Get the base directory and append to sys.path to allow imports
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import streamlit as st
import os
import uuid
from streamlit_javascript import st_javascript
import requests

from agent.lang_graph.graph import AdaptiveRAGGraph
from front_end.utils.message_utils import stream_assistant_response, convert_messages_to_save, summary_conversation_theme

st.set_page_config(layout="wide")

graph = AdaptiveRAGGraph().agent

sidebar_style = """
<style>
[data-testid="stSidebar"] > div:first-child {
    width: 200px;
}
[data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
    width: 200px;
}
[data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
    margin-left: -200px;
}
</style>
"""
st.markdown(sidebar_style, unsafe_allow_html=True)

css_path = os.path.join(os.path.dirname(__file__), "styles", "login_style.css")
if os.path.exists(css_path):
    with open(css_path, "r") as f:
        css_content = f.read()
    st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)

API_URL = "http://localhost:8000"

# Simple session management
if "user_session_id" not in st.session_state:
    st.session_state.user_session_id = None
if "user_name" not in st.session_state:
    st.session_state.user_name = None

# Check for existing session cookie
raw_cookies = st_javascript("document.cookie")
cookies_dict = {}
if raw_cookies:
    for cookie_pair in raw_cookies.split(";"):
        key_value = cookie_pair.strip().split("=")
        if len(key_value) == 2:
            key, value = key_value
            cookies_dict[key] = value

# Get session ID from cookie if available
session_token = cookies_dict.get("session_token")
if session_token and not st.session_state.user_session_id:
    st.session_state.user_session_id = session_token

# Login form
if not st.session_state.user_session_id:
    st.markdown(
        """
        <div class="centered">
            <h1>AI Engineering Q&A</h1>
            <h3>Please login to continue</h3>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    with st.form("login_form"):
        username = st.text_input("Username")
        submit = st.form_submit_button("Login")
        
        if submit and username:
            # Call the backend login API
            response = requests.post(
                f"{API_URL}/auth/login-simple",
                json={"username": username}
            )
            
            if response.status_code == 200:
                # Extract session token from response
                data = response.json()
                new_session_id = data["session_token"]
                st.session_state.user_session_id = new_session_id
                st.session_state.user_name = data["user"]["name"]
                
                # Set cookie via JavaScript
                st.markdown(
                    f"""
                    <script>
                        document.cookie = "session_token={new_session_id}; path=/; max-age=86400";
                    </script>
                    """,
                    unsafe_allow_html=True
                )
                st.rerun()
            else:
                st.error("Failed to login. Please try again.")
    
    st.stop()

# ---------------------- STATES ----------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = None
if "thoughts" not in st.session_state:
    st.session_state.thoughts = ""

# ------------------- Loading Chat History -------------------
def load_conversations():
    session_token = st.session_state.user_session_id
    if session_token:
        resp = requests.get(f"{API_URL}/conversation?session_token={session_token}")
        if resp.status_code == 200:
            data = resp.json()
            return data["conversations"]
        else:
            st.error("Error loading chats.")
            return []
    return []

st.sidebar.title("Conversations")

# Logout button
if st.sidebar.button("Logout"):
    st.session_state.user_session_id = None
    st.session_state.user_name = None
    st.session_state.messages = []
    st.session_state.thread_id = None
    st.session_state.thoughts = ""
    
    # Clear cookie via JavaScript
    st.markdown(
        """
        <script>
            document.cookie = "session_token=; path=/; expires=Thu, 01 Jan 1970 00:00:00 GMT";
        </script>
        """,
        unsafe_allow_html=True
    )
    st.rerun()

if st.sidebar.button("New Chat"):
    st.session_state.messages = []
    st.session_state.thread_id = None
    st.session_state.thoughts = ""
    st.rerun()

# Verifica e exibe o nome do usuário se disponível
if st.session_state.user_name:
    st.sidebar.markdown(f"👤 **{st.session_state.user_name}**")

conversations_list = load_conversations()
if conversations_list:
    for conv in conversations_list:
        label = f"{conv['thread_name']}"
        if st.sidebar.button(label, key=conv["thread_id"]):
            st.session_state.thread_id = conv["thread_id"]
            st.session_state.messages = []
            for role, content in conv["messages"]:
                st.session_state.messages.append({"role": role, "content": content})
            st.rerun()

st.title("AI Engineering Q&A w/ Adaptive RAG")

# ------------------ Exibição Principal --------------------
for msg in st.session_state.messages:
    role = msg["role"]
    content = msg["content"]
    if role == "assistant_thought":
        with st.expander("Model Thoughts", expanded=False):
            st.markdown(content)
    elif role == "assistant_response":
        with st.chat_message("assistant"):
            st.markdown(content)
    elif role == "user":
        with st.chat_message("user"):
            st.markdown(content)
    else:
        with st.chat_message(role):
            st.markdown(content)

prompt = st.chat_input("Chat with me")

if prompt:
    session_token = st.session_state.user_session_id
    if st.session_state.thread_id is None:
        st.session_state.thread_id = (session_token or "") + str(uuid.uuid4())
        conversation_theme = summary_conversation_theme(prompt)
        payload_create = {
            "session_id": session_token,
            "thread_id": st.session_state.thread_id,
            "thread_name": conversation_theme,
            "first_message_role": "user",
            "first_message_content": prompt
        }
        resp = requests.post(f"{API_URL}/conversation", json=payload_create)
        if resp.status_code != 200:
            st.error("Error to create new conversation.")
        st.session_state.messages.append({"role": "user", "content": prompt})
    else:
        # Se já existe
        st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    memory_config = {"configurable": {"thread_id": st.session_state.thread_id}}
    with st.chat_message("assistant"):
        final_response = stream_assistant_response(prompt, graph, memory_config)

    st.session_state.messages.append({"role": "assistant_response", "content": final_response})

    full_msg_objects = graph.get_state(memory_config).values["messages"]

    final_converted = convert_messages_to_save(full_msg_objects)

    update_payload = {
        "thread_id": st.session_state.thread_id,
        "messages": final_converted
    }
    patch_resp = requests.patch(f"{API_URL}/conversation", json=update_payload)
    if patch_resp.status_code != 200:
        st.error("Error on updating conversation.")
