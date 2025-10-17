import sys
from pathlib import Path

import requests
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.config import get_settings
from src.index.embedding import SentenceTransformerEmbedder
from src.ingest.pipeline import IngestionPipeline

st.set_page_config(page_title="RAG Agent", layout="centered")
st.title("RAG Agent")

settings = get_settings()
RAW_DATA_DIR = settings.raw_data_dir
INDEX_DIR = settings.index_dir
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
INDEX_DIR.mkdir(parents=True, exist_ok=True)

st.subheader("1. Upload and Ingest Documents")
uploaded_files = st.file_uploader(
    "Upload PDF/TXT/Markdown files",
    type=["pdf", "txt", "md"],
    accept_multiple_files=True,
)

if uploaded_files:
    for uploaded_file in uploaded_files:
        destination = RAW_DATA_DIR / uploaded_file.name
        destination.write_bytes(uploaded_file.getbuffer())
    st.success(f"Saved {len(uploaded_files)} file(s) to {RAW_DATA_DIR}.")

if st.button("Run Ingestion"):
    try:
        embedder = SentenceTransformerEmbedder(model_name=settings.embedding_model)
        pipeline = IngestionPipeline(index_storage_dir=INDEX_DIR, embedder=embedder)
        pipeline.ingest_directory(RAW_DATA_DIR)
        st.success("Documents ingested successfully.")
    except Exception as exc:
        st.error(f"Ingestion failed: {exc}")

st.divider()
st.subheader("2. Ask Your Question")

q = st.text_input("Question", placeholder="What do the documents say about ...?")

if st.button("Ask") and q.strip():
    try:
        r = requests.get("http://localhost:8000/ask", params={"q": q}, timeout=30)
        r.raise_for_status()
        data = r.json()
        st.subheader("Answer")
        st.write(data["answer"])
        citations = data.get("citations", [])
        if citations:
            st.subheader("Citations")
            for citation in citations:
                with st.expander(f"{citation.get('source', 'unknown source')} — score {citation.get('score', 0):.3f}"):
                    st.write(citation.get("text_preview", ""))
    except Exception as e:
        st.error(f"Request failed: {e}")
