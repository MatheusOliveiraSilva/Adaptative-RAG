# Adaptative-RAG

## Overview

Adaptative-RAG is an advanced Retrieval-Augmented Generation strategy that dynamically adapts to user queries to provide more accurate and contextually relevant responses. It decides to route the user query to one of 3 options:

1. Act as normal chatbot LLM-based, user query don't need extra knowledge.
2. Retrieve document from Pinecone Index (we stored some documents about AI Engineering in index).
3. Web Search for information if we decide that isn't any of the cases above.

## Implementation Architecture

The system is built using a directed graph architecture implemented with LangGraph, allowing for dynamic routing of user queries through different processing paths:

### Core Components

1. **Query Router**: Analyzes incoming user queries and decides whether to use the vector database (for AI engineering topics) or web search for up-to-date information.

2. **Document Retrieval & Grading**: 
   - Retrieves relevant documents from a Pinecone vector store using OpenAI embeddings
   - Grades documents for relevance to the user's question
   - Filters out irrelevant documents to improve response quality

3. **Query Rewriting**: Reformulates queries that don't initially yield quality results to improve document retrieval

4. **Web Search**: Uses Tavily Search API to retrieve real-time information from the web when needed

5. **Response Generation**: Uses Claude 3.7 Sonnet to generate comprehensive, accurate responses based on the retrieved context

6. **Response Validation**: Checks responses for hallucinations and relevance to ensure high-quality answers

### Technology Stack

- **Vector Database**: Pinecone for document storage and retrieval
- **Embedding Models**: OpenAI's text-embedding-3-large for document vectorization
- **LLMs**: 
  - Claude 3.7 Sonnet for high-quality response generation (final generation)
  - GPT-4o-mini for routing and grading tasks (fast on latency)
- **Frontend**: Streamlit-based interface for user interaction
- **Processing Framework**: LangGraph for flexible workflow orchestration

### Decision Flow

1. User submits a question via the Streamlit interface
2. The system analyzes the question to determine the best information source
3. Based on the decision, the system either:
   - Retrieves documents from Pinecone and grades their relevance
   - Searches the web for current information
4. The system generates a response using the retrieved information
5. The response is checked for quality and relevance
6. If needed, the query is reformulated and the process repeats
7. The final response is streamed to the user

This adaptive approach ensures that responses are not only accurate but also sourced from the most appropriate knowledge base for each specific query.

You can check out the application at [my site](https://adaptative-rag-matheusolivsilv.streamlit.app/) deployed with [Streamlit Cloud](https://share.streamlit.io/).

### Running locally:

To run this application on your local machine, follow these steps:

#### Prerequisites

- Python 3.11+ installed
- Git installed
- API keys for:
  - OpenAI (for embeddings and GPT-4o-mini)
  - Anthropic (for Claude 3.7 Sonnet)
  - Pinecone (for vector database)
  - Tavily (for web search)

#### Setup Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/MatheusOliveiraSilva/Adaptative-RAG.git
   cd Adaptative-RAG
   ```

2. **Create and activate a virtual environment**
   ```bash
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
4. **Run my chatbot memory API in background to manage sessions and chat history.** 

    To understand how i manage the chat history of users, check this [repository](https://github.com/MatheusOliveiraSilva/ChatMemoryAPI).

4. **Set up environment variables**
   
   Create a `.env` like the `.env_example` file in the root directory with the following variables.

6. **Initialize the vector database (first time only)**
   
   If is your first time running this project, you need to send to pinecone at least the basic documents we selected, to do this check [pinecone initialization file](agent/preprocessment/load_web_page.py). Run it with:
   ```bash
   python -m agent.preprocessment.load_web_page
   ```
   
   This will create a Pinecone index and populate it with AI engineering related content.

#### Running the Application

1. **Start the Streamlit frontend**
   ```bash
   streamlit run front_end/main_page.py
   ```

2. **Access the application**
   
   Open your browser and navigate to:
   ```
   http://localhost:8501
   ```

#### Customization

You can customize the behavior of the system by modifying:
- Prompts in `agent/lang_graph/prompts.py`
- Decision logic in `agent/lang_graph/edges.py`
- Processing nodes in `agent/lang_graph/nodes.py`
- UI components in `front_end/main_page.py`

