import os
from typing import List
from langchain_core.documents import Document
from app.loaders.file_loader import FileLoader
from app.processing.chunk_builder import ChunkBuilder
from app.processing.multimodal_extractor import MultimodalExtractor
from app.vectorstore.chroma_store import ChromaVectorStore
from app.rag.rag_pipeline import MultimodalRAG

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import json


class MultimodalRAGOrchestrator:
    """High-level pipeline controller with AI-enhanced summaries and file hash support."""

    def __init__(self, pdf_path: str):
        """Create an orchestrator for a specific PDF file path.

        Args:
            pdf_path: Path to the PDF file this orchestrator will operate on.

        The orchestrator wires together the file loader, chunk builder,
        multimodal extractor, vector store, and LLM.
        """
        self.loader = FileLoader(pdf_path)
        self.chunker = ChunkBuilder()
        self.extractor = MultimodalExtractor()
        self.vector_store = ChromaVectorStore()
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0)

    def ingest(self, file_hash: str = None) -> None:
        """
        Load PDF, create chunks, generate AI-enhanced summaries, and add to vector DB.
        Stores file_hash in metadata to allow skipping already ingested PDFs.
        """
        elements = self.loader.load_pdf()
        chunks = self.chunker.build(elements)

        documents: List[Document] = []
        total_chunks = len(chunks)

        for i, chunk in enumerate(chunks):
            print(f"Processing chunk {i+1}/{total_chunks}...")

            content_data = self.extractor.extract(chunk)
            print(f"  Found types: {content_data['types']}")
            print(f"  Tables: {len(content_data['tables'])}, Images: {len(content_data['images'])}")

            # Create AI-enhanced summary if tables/images exist
            if content_data['tables'] or content_data['images']:
                enhanced_content = self.create_ai_enhanced_summary(
                    content_data['text'],
                    content_data['tables'],
                    content_data['images']
                )
            else:
                enhanced_content = content_data['text']

            # Serialize complex metadata as JSON string
            doc = Document(
                page_content=enhanced_content,
                metadata={
                    "original_content": json.dumps({
                        "raw_text": content_data['text'],
                        "tables_html": content_data['tables'],
                        "images_base64": content_data['images']
                    }),
                    "file_name": os.path.basename(self.loader.file_path)
                }
            )
            documents.append(doc)

        # Add documents to Chroma with file_hash
        self.vector_store.add_documents(documents, file_hash=file_hash)
        print(f"✅ Ingested {len(documents)} chunks into vector DB.")

    def create_ai_enhanced_summary(self, text: str, tables: List[str], images: List[str]) -> str:
        """Send chunk content to LLM for an enhanced, searchable summary."""
        try:
            prompt_text = f"""You are creating a searchable description for document content retrieval.

            TEXT CONTENT:
            {text}

        """
            if tables:
                prompt_text += "TABLES:\n"
                for i, table in enumerate(tables):
                    prompt_text += f"Table {i+1}:\n{table}\n\n"

                prompt_text += """
                        YOUR TASK:
                        Generate a comprehensive, searchable description that covers:

                        1. Key facts, numbers, and data points from text and tables
                        2. Main topics and concepts discussed
                        3. Questions this content could answer
                        4. Visual content analysis (charts, diagrams, patterns in images)
                        5. Alternative search terms users might use

                        Make it detailed and searchable - prioritize findability over brevity.

                        SEARCHABLE DESCRIPTION:"""

            # Prepare messages for LLM
            message_content = [{"type": "text", "text": prompt_text}]
            for image_base64 in images:
                message_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
                })

            message = HumanMessage(content=message_content)
            response = self.llm.invoke([message])
            return response.content

        except Exception as e:
            print(f"❌ AI summary failed: {e}")
            summary = f"{text[:300]}..."
            if tables:
                summary += f" [Contains {len(tables)} table(s)]"
            if images:
                summary += f" [Contains {len(images)} image(s)]"
            return summary

    def get_rag(self) -> MultimodalRAG:
        """Return a MultimodalRAG instance wired to this orchestrator's vector store.

        The returned RAG is constructed with a retriever created from the
        underlying Chroma vector store. This makes it ready to accept query
        strings and produce LLM-backed answers using retrieved context.
        """
        return MultimodalRAG(self.vector_store.as_retriever())
