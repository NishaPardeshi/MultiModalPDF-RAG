import os
from typing import List
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings


class ChromaVectorStore:
    def __init__(
        self,
        persist_dir: str = "./chroma_store",
        collection_name: str = "multimodal_rag",
    ):
        # Ensure the persist directory exists and is writable
        if not os.path.exists(persist_dir):
            os.makedirs(persist_dir, exist_ok=True)
        if not os.access(persist_dir, os.W_OK):
            raise PermissionError(f"Chroma persist directory '{persist_dir}' is not writable")

        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small"
        )

        self.store = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=persist_dir,
        )

    # ---------------------------
    # Check if file exists
    # ---------------------------
    def document_exists(self, file_hash: str) -> bool:
        results = self.store.get(where={"file_hash": file_hash}, limit=1)
        return len(results["ids"]) > 0

    # ---------------------------
    # Add documents
    # ---------------------------
    def add_documents(self, documents: list, file_hash: str = None) -> None:
        # Add optional file_hash metadata to each doc
        if file_hash:
            for doc in documents:
                doc.metadata["file_hash"] = file_hash
                doc.metadata["file_name"] = doc.metadata.get("file_name", "unknown.pdf")
        self.store.add_documents(documents)

    # ---------------------------
    # As retriever
    # ---------------------------
    def as_retriever(self, k: int = 5):
        return self.store.as_retriever(search_kwargs={"k": k})

    # ---------------------------
    # List all ingested PDFs
    # ---------------------------
    def list_documents(self) -> list:
        """Return a list of PDF file names already ingested."""
        try:
            all_docs = self.store.get(where={}, limit=1000)
            pdf_names = []
            for md in all_docs["metadatas"]:
                file_name = md.get("file_name")
                if file_name and file_name not in pdf_names:
                    pdf_names.append(file_name)
            return pdf_names
        except Exception:
            return []
