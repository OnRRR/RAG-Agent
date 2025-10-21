import base64
import sys
from html import escape
from pathlib import Path
from typing import List

import requests
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.config import get_settings
from src.index.embedding import SentenceTransformerEmbedder
from src.ingest.pipeline import IngestionPipeline

st.set_page_config(page_title="RAG Agent", layout="wide")

CUSTOM_CSS = """
<style>
body {
    background: #f4f5fb;
}
section.main > div {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}
.page-title {
    font-size: 2rem;
    font-weight: 700;
    color: #1f2933;
}
.page-subtitle {
    color: #52606d;
    font-size: 1rem;
    margin-bottom: 1.75rem;
}
.glass-card {
    background: rgba(255, 255, 255, 0.95);
    border-radius: 22px;
    padding: 1.75rem 1.75rem 1.5rem;
    box-shadow: 0 18px 44px rgba(15, 23, 42, 0.12);
    border: 1px solid rgba(148, 163, 184, 0.14);
    backdrop-filter: blur(14px);
}
.glass-card + .glass-card {
    margin-top: 1.5rem;
}
.doc-metadata {
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 0.85rem;
    color: #64748b;
    margin-bottom: 0.75rem;
}
.doc-preview {
    border-radius: 18px;
    overflow: hidden;
    border: 1px solid rgba(148, 163, 184, 0.18);
}
.chat-wrapper {
    display: flex;
    flex-direction: column;
    height: 88vh;
    background: rgba(255, 255, 255, 0.95);
    border-radius: 22px;
    padding: 1.75rem 1.5rem 1.25rem;
    box-shadow: 0 18px 44px rgba(15, 23, 42, 0.12);
    border: 1px solid rgba(148, 163, 184, 0.14);
    backdrop-filter: blur(14px);
}
.chat-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 1rem;
}
.chat-scroll {
    flex: 1;
    overflow-y: auto;
    padding-right: 0.75rem;
}
.chat-scroll::-webkit-scrollbar {
    width: 6px;
}
.chat-scroll::-webkit-scrollbar-thumb {
    background: rgba(148, 163, 184, 0.55);
    border-radius: 999px;
}
.message-assistant,
.message-user {
    max-width: 92%;
    padding: 0.95rem 1.15rem;
    border-radius: 18px;
    margin-bottom: 0.85rem;
    line-height: 1.6;
    font-size: 0.97rem;
    box-shadow: 0 10px 24px rgba(15, 23, 42, 0.1);
}
.message-assistant {
    background: #f1f5f9;
    color: #0f172a;
    border-bottom-left-radius: 6px;
}
.message-user {
    margin-left: auto;
    background: linear-gradient(135deg, #6366f1, #4f46e5);
    color: #f9fafb;
    border-bottom-right-radius: 6px;
}
.message-meta {
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.35rem;
    opacity: 0.7;
}
.placeholder {
    text-align: center;
    padding: 3rem 1rem;
    color: #64748b;
    border: 1px dashed rgba(148, 163, 184, 0.4);
    border-radius: 18px;
    margin-bottom: 1.5rem;
}
.clear-button button {
    background: rgba(244, 63, 94, 0.12);
    color: #be123c;
    border: none;
}
.clear-button button:hover {
    background: rgba(244, 63, 94, 0.18);
    color: #9f1239;
}
.primary-button button {
    background: linear-gradient(135deg, #6366f1, #4f46e5);
    color: #f9fafb;
    border: none;
}
.primary-button button:hover {
    background: linear-gradient(135deg, #4f46e5, #4338ca);
}
.chat-footer {
    padding-top: 1.25rem;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

settings = get_settings()
RAW_DATA_DIR = settings.raw_data_dir
INDEX_DIR = settings.index_dir
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
INDEX_DIR.mkdir(parents=True, exist_ok=True)

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}


def list_documents() -> List[Path]:
    return sorted(
        [doc for doc in RAW_DATA_DIR.glob("*") if doc.is_file() and doc.suffix.lower() in SUPPORTED_EXTENSIONS],
        key=lambda path: path.name.lower(),
    )


def render_pdf(path: Path, *, height: int = 720) -> None:
    with path.open("rb") as handle:
        base64_pdf = base64.b64encode(handle.read()).decode("utf-8")
    st.markdown(
        f'<iframe src="data:application/pdf;base64,{base64_pdf}#view=FitH" width="100%" height="{height}" '
        'type="application/pdf"></iframe>',
        unsafe_allow_html=True,
    )


def render_text(path: Path, *, height: int = 720) -> None:
    content = path.read_text(encoding="utf-8", errors="replace")
    st.text_area("Document preview", value=content, height=height, label_visibility="collapsed")


def render_document(doc_path: Path, *, height: int = 720) -> None:
    if doc_path.suffix.lower() == ".pdf":
        render_pdf(doc_path, height=height)
    else:
        render_text(doc_path, height=height)


def format_text(text: str) -> str:
    return escape(text).replace("\n", "<br>")


def render_message(role: str, text: str) -> None:
    css_class = "message-user" if role == "user" else "message-assistant"
    label = "You" if role == "user" else "Assistant"
    st.markdown(
        f"""
        <div class="{css_class}">
            <div class="message-meta">{label}</div>
            <div class="message-body">{format_text(text)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


st.markdown('<div class="page-title">Library Copilot</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="page-subtitle">Browse research papers on the left and chat with your assistant on the right.</div>',
    unsafe_allow_html=True,
)

col_docs, col_chat = st.columns([1.35, 1], gap="large")

with col_docs:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("Document Library", anchor=False)
    st.caption("Upload new files and rebuild the knowledge index when ready.")
    uploaded_files = st.file_uploader(
        "Upload PDF/TXT/Markdown files",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
        label_visibility="collapsed",
        key="uploader",
    )

    if uploaded_files:
        saved = 0
        for uploaded_file in uploaded_files:
            destination = RAW_DATA_DIR / uploaded_file.name
            destination.write_bytes(uploaded_file.getbuffer())
            saved += 1
        try:
            relative_dir = RAW_DATA_DIR.relative_to(ROOT_DIR)
        except ValueError:
            relative_dir = RAW_DATA_DIR
        st.success(f"Saved {saved} file(s) to {relative_dir}.")

    st.caption("Once uploads finish, rebuild the embeddings to include them in search.")
    with st.container():
        if st.button("Rebuild Index", type="primary", use_container_width=True, key="rebuild"):
            try:
                embedder = SentenceTransformerEmbedder(model_name=settings.embedding_model)
                pipeline = IngestionPipeline(index_storage_dir=INDEX_DIR, embedder=embedder)
                pipeline.ingest_directory(RAW_DATA_DIR)
                st.success("Documents ingested successfully.")
            except Exception as exc:
                st.error(f"Ingestion failed: {exc}")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("Document Viewer", anchor=False)
    documents = list_documents()
    if documents:
        selected_name = st.selectbox(
            "Select a document",
            options=[doc.name for doc in documents],
            index=0,
            label_visibility="collapsed",
            key="doc_select",
        )
        selected_doc = next(doc for doc in documents if doc.name == selected_name)
        stats = selected_doc.stat()
        st.markdown(
            f'<div class="doc-metadata"><span>{selected_doc.name}</span><span>{stats.st_size / 1024:.1f} KB</span></div>',
            unsafe_allow_html=True,
        )
        st.markdown('<div class="doc-preview">', unsafe_allow_html=True)
        render_document(selected_doc, height=720)
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.markdown(
            '<div class="placeholder">Upload a PDF or text file to preview it here.</div>',
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

with col_chat:
    st.markdown('<div class="chat-wrapper">', unsafe_allow_html=True)
    st.markdown(
        '<div class="chat-header"><h3 style="margin-bottom:0;">Research Assistant</h3>',
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="chat-scroll">', unsafe_allow_html=True)
    if not st.session_state.chat_history:
        st.markdown(
            '<div class="placeholder">Ask a question about the open document to begin the conversation.</div>',
            unsafe_allow_html=True,
        )
    else:
        for exchange in st.session_state.chat_history:
            render_message("user", exchange["question"])
            render_message("assistant", exchange["answer"])
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="chat-footer">', unsafe_allow_html=True)
    columns = st.columns([0.75, 0.25])
    with columns[0]:
        st.caption("Ask your question to receive a grounded answer.")
    with columns[1]:
        if st.button("Clear", key="clear_chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()
    prompt = st.chat_input("Ask the assistant anything about your documents")
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    if prompt:
        try:
            response = requests.get("http://localhost:8000/ask", params={"q": prompt}, timeout=60)
            response.raise_for_status()
            payload = response.json()
            st.session_state.chat_history.append(
                {
                    "question": prompt,
                    "answer": payload.get("answer", "No answer returned."),
                }
            )
            st.rerun()
        except Exception as exc:
            st.error(f"Request failed: {exc}")
