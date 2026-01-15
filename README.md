# Multimodal RAG 

<img width="1536" height="1024" alt="MultiModalPDF-RAG" src="https://github.com/user-attachments/assets/260fd9c8-4744-4767-8a4b-065a14d935d2" />

## Demo

https://github.com/user-attachments/assets/ff65ae36-981e-43a7-99fe-95d77ba5751a

## Project summary

This repository contains a small experimental Retrieval-Augmented Generation
(RAG) pipeline that ingests PDFs, extracts text/tables/images, creates
embeddings (OpenAI), stores them in a Chroma collection, and exposes a
Gradio-based web UI for asking questions over ingested documents.

Key features:
- PDF partitioning (via Unstructured) with table and image extraction
- Title-aware chunking for better retrieval granularity
- Optional AI-enhanced chunk summaries (uses LLM)
- Embeddings stored in Chroma for fast retrieval
- Gradio chat UI with streaming and progress feedback

## Files of interest

- `gradio_app.py` — Main Gradio app (upload + chat handlers).
- `app/orchestrator.py` — High-level pipeline: loader → chunker → extractor → vector store → LLM.
- `app/loaders/file_loader.py` — Loads and partitions PDFs.
- `app/processing/` — Chunking (`chunk_builder.py`) and multimodal extraction (`multimodal_extractor.py`).
- `app/vectorstore/chroma_store.py` — Thin wrapper around Chroma for add/query/list operations.
- `app/rag/rag_pipeline.py` — RAG runtime: retrieves context and calls the LLM (supports streaming).
- `uploads/` — Where uploaded PDFs are saved.
- `chroma_store/` — Directory where Chroma persists data (SQLite and binary blobs).

## Architecture (diagram)

Below is a high-level architecture diagram showing components and data flow. The Mermaid diagram will render on GitHub and many Markdown viewers.

Gradio UI -> Orchestrator -> {FileLoader, ChunkBuilder -> MultimodalExtractor} -> ChromaVectorStore -> Chroma files
																	 \-> MultimodalRAG -> LLM/Embeddings


## API reference (key functions / methods)

This project is a small codebase without a formal HTTP API — the "API" here
is the Python functions and methods you call from `gradio_app.py` or a REPL.

- `gradio_app.get_file_hash(file_path: str) -> str` — Compute SHA-256 hash for a file.
- `gradio_app.upload_pdf(file, progress) -> (str, Optional[str])` — Save upload, check dedupe, and ingest into vector store. Returns (status_message, saved_path).
- `gradio_app.chat_with_pdf(message, history, pdf_path)` — Generator (streaming) or function (non-streaming variant) that queries the RAG and yields/returns chat updates.

- `app.loaders.FileLoader(file_path)`
	- `load_pdf() -> List[Element]` — Partition a PDF into unstructured elements (text blocks, tables, images).

- `app.processing.ChunkBuilder(...)`
	- `build(elements: List[Element]) -> List[Chunk]` — Create title-aware chunks from parsed elements.

- `app.processing.MultimodalExtractor.extract(chunk) -> Dict` — Extracts `text`, list of `tables` (HTML), `images` (base64), and `types` detected in the chunk.

- `app.vectorstore.ChromaVectorStore(...)`
	- `add_documents(documents: List[Document], file_hash: Optional[str])` — Add LangChain `Document` objects to Chroma. Attaches `file_hash` metadata if provided.
	- `document_exists(file_hash: str) -> bool` — Returns True if at least one doc with the given hash exists.
	- `as_retriever(k: int=5)` — Return a LangChain retriever configured to return up to `k` documents.
	- `list_documents() -> List[str]` — (Utility) Return a deduplicated list of uploaded PDF filenames previously ingested.

- `app.rag.MultimodalRAG(retriever)`
	- `query(question: str) -> str or generator` — Retrieve relevant documents and call the LLM to synthesize an answer. May stream tokens when LLM streaming is enabled.

## Example outputs

1) Ingestion logs (what you might see on the console during `ingest()`):

```
Processing chunk 1/12...
	Found types: ['text']
Processing chunk 2/12...
	Found types: ['text', 'table']
Processing chunk 3/12...
	Found types: ['text', 'image']
✅ Ingested 12 chunks into vector DB.
```

2) Example document metadata stored with a `Document` (serialized JSON inside `original_content`):

```json
{
	"raw_text": "This chapter describes the network architecture...",
	"tables_html": ["<table>...</table>"],
	"images_base64": ["/9j/4AAQSkZJRgABAQAAAQABAAD..."]
}
```

3) Example chat session (user -> assistant):

User: "What is the main contribution of the paper?"

Assistant: "The paper introduces a title-aware chunking strategy that preserves
section boundaries for improved retrieval. It demonstrates better recall on
table-based queries and includes an AI-enhanced summary for image and table
content."

4) Streaming example (how UI updates while model streams tokens):

- Immediately after sending: history shows the user's message.
- As the model streams tokens: assistant message content grows progressively,
	e.g. "The paper introduces a title-aware..." -> "The paper introduces a title-aware chunking strategy..."

## Existing sections (setup, run, troubleshooting)

See earlier sections above for setup and quick-run instructions. In short:

```bash
python gradio_app.py
```

Then upload a PDF and ask questions in the Gradio UI.





