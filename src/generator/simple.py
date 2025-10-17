from __future__ import annotations

from typing import Iterable, List

from src.generator.types import Citation, GeneratedAnswer
from src.retriever.faiss_retriever import RetrievedChunk


class EchoGenerator:
    """Placeholder generator that stitches together retrieved chunks.

    Replace this with a real LLM-backed generator once credentials/models are configured.
    """

    def generate(self, query: str, chunks: Iterable[RetrievedChunk]) -> GeneratedAnswer:
        chunk_list: List[RetrievedChunk] = list(chunks)
        if not chunk_list:
            return GeneratedAnswer(answer="I couldn't find relevant information in the indexed documents.", citations=[])

        context_preview = "\n\n".join(f"- {chunk.text[:300]}" for chunk in chunk_list)
        answer = (
            "This is a placeholder response. Configure an LLM to produce grounded answers.\n\n"
            f"User question: {query}\n\n"
            "Top retrieved snippets:\n"
            f"{context_preview}"
        )
        citations = [
            Citation(
                chunk_id=chunk.chunk_id,
                source=chunk.metadata.get("source"),
                score=chunk.score,
                text_preview=chunk.text[:200],
            )
            for chunk in chunk_list
        ]
        return GeneratedAnswer(answer=answer, citations=citations)
