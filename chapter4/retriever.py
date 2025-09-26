import os
import tempfile
from typing import Any

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

from chapter4.document_loader import load_document
from chapter4.llms import EMBEDDINGS

VECTOR_STORE = InMemoryVectorStore(embedding=EMBEDDINGS)


def split_document(docs: list[Document]) -> list[Document]:
    """Split each document"""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    return text_splitter.split_documents(docs)


class DocumentRetriever(BaseRetriever):
    """A retriever that contains the top k documents that contains in the user query."""

    documents: list[Document] = []
    k: int = 5

    def model_post_init(self, ctx: Any) -> None:
        self.store_documents(self.documents)

    @staticmethod
    def store_documents(docs: list[Document]) -> None:
        """Add documents to the vector store."""
        splits = split_document(docs)
        VECTOR_STORE.add_documents(splits)

    def add_uploaded_docs(self, uploaded_files: list[Any]) -> None:
        """Add uploaded documents."""
        docs = []
        with tempfile.TemporaryDirectory() as temp_dir:
            for file in uploaded_files:
                temp_filepath = os.path.join(temp_dir, file.name)
                with open(temp_filepath, "wb") as f:
                    f.write(file.getvalue())
                try:
                    docs.extend(load_document(temp_filepath))
                except Exception as e:
                    print(f"Fail to load {file.name}: {e}")
        self.documents.extend(docs)
        self.store_documents(docs)

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        """Sync implementation for retriever."""
        if len(self.documents) == 0:
            return []
        return VECTOR_STORE.similarity_search(query=query, k=self.k)
