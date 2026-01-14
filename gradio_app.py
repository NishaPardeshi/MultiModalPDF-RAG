import gradio as gr
import os
import time
import hashlib
from dotenv import load_dotenv

from app.orchestrator import MultimodalRAGOrchestrator
from app.vectorstore.chroma_store import ChromaVectorStore

load_dotenv()

UPLOAD_DIR = "./uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

vector_store = ChromaVectorStore()

# ---------------------------
# Utils
# ---------------------------

def get_file_hash(file_path: str) -> str:
    """Compute the SHA-256 hash for a file.

    Args:
        file_path: Path to the file to hash.

    Returns:
        Hexadecimal SHA-256 digest of the file contents.
    """
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        hasher.update(f.read())
    return hasher.hexdigest()


# ---------------------------
# PDF Upload & Ingestion
# ---------------------------

def upload_pdf(file, progress=gr.Progress(track_tqdm=True)):
    """Handle an uploaded PDF: save, check, and ingest if new.

    This function is wired to the Gradio "Ingest PDF" button. It writes the
    uploaded temporary file to the project's upload directory, computes a
    content hash, and then uses the orchestrator pipeline to ingest the
    document into the vector store unless it was already ingested.

    Args:
        file: File-like object provided by Gradio File component.

    Returns:
        A tuple of (status_message, saved_file_path_or_None).
    """
    if file is None:
        return "❌ Please upload a PDF", None

    # Restricting file size to 2.5 MB
    MAX_FILE_SIZE_MB = 2.5
    MAX_FILE_SIZE_BYTES = int(MAX_FILE_SIZE_MB * 1024 * 1024)

    # Validate file size
    file_size = os.path.getsize(file.name)
    if file_size > MAX_FILE_SIZE_BYTES:
        return (
            f"❌ File too large ({file_size / (1024*1024):.2f} MB). "
            f"Maximum allowed size is {MAX_FILE_SIZE_MB} MB.",
            None
        )
    
    # Save the file to UPLOAD_DIR
    file_path = os.path.join(UPLOAD_DIR, os.path.basename(file.name))
    os.replace(file.name, file_path)

    file_hash = get_file_hash(file_path)
    progress(0.1, desc="Checking if PDF already exists...")
    time.sleep(0.4)

    if vector_store.document_exists(file_hash):
        return "📄 PDF already ingested. Ready to chat.", file_path
    
    progress(0.3, desc="Parsing PDF...")

    pipeline = MultimodalRAGOrchestrator(file_path)

    progress(0.6, desc="Generating embeddings & summaries...")
    pipeline.ingest(file_hash=file_hash)

    progress(1.0, desc="Done")
    return "✅ PDF ingested successfully!", file_path


# ---------------------------
# Chat Function
# ---------------------------

def chat_with_pdf(message, history, pdf_path):
    """Stream an answer to the Gradio chat UI for a selected PDF.

    This generator function yields incremental updates so the Gradio UI
    can display the user's message immediately and stream assistant
    tokens as they are produced by the RAG+LLM pipeline.

    Args:
        message: The user's message/question.
        history: Current chat history in Gradio messages format.
        pdf_path: Path to the ingested PDF used to create the RAG.

    Yields:
        A tuple of (history, textbox_value) where history is the updated
        messages list and textbox_value is the content of the input box
        (usually empty string to clear the textbox).
    """
    if history is None:
        history = []

    if not pdf_path:
        history.append({"role": "assistant", "content": "❌ Please select a PDF first."})
        yield history, ""
        return 
    
    # Add user message immediately
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": ""})

    yield history, ""  # show user message instantly

    pipeline = MultimodalRAGOrchestrator(pdf_path)
    rag = pipeline.get_rag()

    assistant_text = ""
    for token in rag.query(message):
        assistant_text += token
        history[-1]["content"] = assistant_text
        yield history, ""  # stream update

# ---------------------------
# UI Layout
# ---------------------------

with gr.Blocks(
    title="📄 Multimodal RAG",
    theme=gr.themes.Soft(primary_hue="violet")
    ) as demo:
    
    gr.Markdown(
        """
        # 📄 Multimodal RAG Chat
        Ask questions across all uploaded PDFs
        """
    )

    pdf_state = gr.State(None)

    # Upload row
    with gr.Row(equal_height=True):

        # ---------- LEFT: Upload ----------
        with gr.Column(scale=1):
            gr.Markdown("### ➕ Upload PDF to Ingest")
            pdf_upload = gr.File(file_types=[".pdf"])
            ingest_btn = gr.Button("🚀 Ingest PDF", variant="primary")
            upload_status = gr.Markdown()


        # ---------- CENTER: Chat ----------
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(height=450)

            with gr.Row():
                msg = gr.Textbox(
                    placeholder="Ask questions about the PDFs...",
                    scale=4
                )
                send_btn = gr.Button("Send", scale=1)

        # --------- Events ---------
        ingest_btn.click(
            upload_pdf,
            inputs=pdf_upload,
            outputs=[upload_status, pdf_state],
            show_progress=True
        )

        send_btn.click(
            chat_with_pdf,
            inputs=[msg, chatbot, pdf_state],
            outputs=[chatbot, msg]
        )

demo.launch()

