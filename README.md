# Adaptative-RAG

## Overview

Adaptative-RAG is an advanced Retrieval-Augmented Generation system that dynamically adapts to user queries to provide more accurate and contextually relevant responses. By combining the power of large language models with efficient document retrieval techniques, this system enhances the quality of AI-generated responses by grounding them in factual information.

## Module Structure

The repository is organized into the following modules:

1. **API Layer** - Handles HTTP requests and serves as the interface between clients and the core RAG system.
2. **Retrieval Engine** - Responsible for document indexing, vector storage, and efficient retrieval of relevant context.
3. **Adaptation Module** - Dynamically adjusts retrieval strategies based on query characteristics and feedback.
4. **Generation Module** - Integrates with language models to produce coherent and factually accurate responses.
5. **Frontend** - Provides a user-friendly interface for interacting with the RAG system.

## Getting Started

### Setting up the Environment

1. Create a virtual environment:
   ```bash
   python -m venv venv
   ```

2. Activate the virtual environment:
   - On Windows:
     ```bash
     venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source venv/bin/activate
     ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application

1. Start the API server:
   ```bash
   python api/app.py
   ```

2. Launch the frontend application:
   ```bash
   cd frontend
   npm install  # Only needed for the first time
   npm start
   ```

3. Access the application in your browser at `http://localhost:3000`