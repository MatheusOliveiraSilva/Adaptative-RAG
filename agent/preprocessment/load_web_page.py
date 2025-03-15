import os
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from uuid import uuid4
from dotenv import load_dotenv
from pathlib import Path

root_dir = Path().absolute()

load_dotenv(root_dir / ".env")

class WebPageLoader:
    """
    Class to load web pages and add them to a vector store.
    """

    def __init__(self, index_name: str, urls: list[str]):
        self.urls = urls

        self.pc = Pinecone(
            api_key=os.getenv("PINECONE_API_KEY")
        )

        self.embedding_model = OpenAIEmbeddings(
            model="text-embedding-3-large",
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )

        self.pc.create_index(
            name=index_name,
            dimension=3072,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )

        self.index = self.pc.Index(index_name)
        self.vector_store = PineconeVectorStore(index=self.index, embedding=self.embedding_model)

        self.docs = self.load_web_pages(urls)

        self.chunked_docs = self.chunk_docs(self.docs)

        self.add_docs_to_vector_store(self.chunked_docs)

    def add_docs_to_vector_store(self, docs: list[Document]) -> None:
        uuids = [str(uuid4()) for _ in range(len(docs))]

        self.vector_store.add_documents(documents=docs, ids=uuids)

    def load_web_pages(self, urls: list[str]) -> list[Document]:
        docs = [WebBaseLoader(url).load() for url in urls]
        docs_list = [item for sublist in docs for item in sublist]

        return docs_list

    def chunk_docs(self, docs: list[Document]) -> list[Document]:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,
            chunk_overlap=128
        )

        return text_splitter.split_documents(docs)

if __name__ == "__main__":
    urls = [
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
        "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
        "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
    ]

    WebPageLoader(index_name="web-ai-engineer-index", urls=urls)
