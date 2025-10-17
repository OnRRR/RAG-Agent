from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from src.config import get_settings
from src.generator.rag import RAGPipeline

app = FastAPI(title="RAG Agent API")


class Citation(BaseModel):
    chunk_id: str
    source: str | None = None
    score: float
    text_preview: str


class AskResponse(BaseModel):
    answer: str
    citations: list[Citation]


settings = get_settings()
_pipeline: RAGPipeline | None = None


def get_pipeline() -> RAGPipeline:
    global _pipeline
    if _pipeline is None:
        try:
            _pipeline = RAGPipeline.from_settings(settings)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail="Index not available. Please ingest documents first.") from exc
    return _pipeline


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/ask", response_model=AskResponse)
def ask(q: str = Query(..., min_length=3)):
    pipeline = get_pipeline()
    result = pipeline.answer(q)
    return AskResponse(
        answer=result.answer,
        citations=[
            Citation(
                chunk_id=citation.chunk_id,
                source=citation.source,
                score=citation.score,
                text_preview=citation.text_preview,
            )
            for citation in result.citations
        ],
    )
