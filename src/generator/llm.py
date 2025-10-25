from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterable, List

import requests

from src.config import Settings
from src.generator.types import Citation, GeneratedAnswer
from src.retriever.faiss_retriever import RetrievedChunk
from src.utils.logger import get_logger

logger = get_logger(__name__)


SYSTEM_PROMPT = (
    "You are RAG-Agent, an AI specialist tasked with answering questions using the supplied document excerpts only. "
    "Follow these rules:\n"
    "1. Read the provided context carefully and gather only the facts that directly relate to the question.\n"
    "2. Synthesize a concise answer grounded in those facts. If multiple relevant points exist, summarise them clearly.\n"
    "3. If the context does not contain enough information to answer with confidence, explicitly respond with "
    "'I am not sure based on the indexed documents.' and offer a brief note about what is missing.\n"
    "4. Do not invent, assume, or import knowledge that is not present in the context."
)


@dataclass(slots=True)
class LLMGenerator:
    settings: Settings

    def generate(self, query: str, chunks: Iterable[RetrievedChunk]) -> GeneratedAnswer:
        chunk_list: List[RetrievedChunk] = list(chunks)
        if not chunk_list:
            return GeneratedAnswer(
                answer="I couldn't find relevant information in the indexed documents.",
                citations=[],
            )

        provider = self.settings.llm_provider
        logger.info("Generating answer via LLM provider '%s'", provider)
        try:
            if provider == "openai":
                answer = self._call_openai(query, chunk_list)
            elif provider == "ollama":
                answer = self._call_ollama(query, chunk_list)
            else:
                answer = self._fallback_answer(query, chunk_list)
        except Exception as exc:  # pylint: disable=broad-except
            logger.error("LLM call failed: %s", exc)
            answer = self._fallback_answer(query, chunk_list, error=str(exc))

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

    def _select_context(self, chunks: List[RetrievedChunk]) -> str:
        max_chunks = max(1, self.settings.llm_max_context_chunks)
        selected = chunks[:max_chunks]
        formatted = []
        for chunk in selected:
            source = chunk.metadata.get("source", "unknown")
            title = chunk.metadata.get("title") or chunk.metadata.get("section_title") or "unknown"
            section = chunk.metadata.get("section") or chunk.metadata.get("section_title") or "section"
            page_start = chunk.metadata.get("page_start")
            page_end = chunk.metadata.get("page_end")
            page_info = ""
            if page_start and page_end:
                page_info = f" (pages {page_start}-{page_end})"
            elif page_start:
                page_info = f" (page {page_start})"
            formatted.append(
                f"Document: {source}\nTitle: {title}\nSection: {section}{page_info}\nSnippet:\n{chunk.text.strip()}"
            )
        return "\n\n".join(formatted)

    def _call_openai(self, query: str, chunks: List[RetrievedChunk]) -> str:
        if not self.settings.llm_openai_api_key:
            raise RuntimeError("OpenAI API key not configured. Set RAG_LLM_OPENAI_API_KEY.")

        base_url = (self.settings.llm_openai_base_url or "https://api.openai.com/v1").rstrip("/")
        url = f"{base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.settings.llm_openai_api_key}",
            "Content-Type": "application/json",
        }
        context = self._select_context(chunks)
        payload = {
            "model": self.settings.llm_openai_model,
            "temperature": self.settings.llm_openai_temperature,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Question:\n{query}\n\nContext:\n{context}\n\nAnswer using only this context.",
                },
            ],
        }
        response = requests.post(url, headers=headers, json=payload, timeout=120)
        response.raise_for_status()
        body = response.json()
        return body["choices"][0]["message"]["content"].strip()

    def _call_ollama(self, query: str, chunks: List[RetrievedChunk]) -> str:
        base_url = self.settings.llm_ollama_base_url.rstrip("/")
        url = f"{base_url}/api/generate"
        context = self._select_context(chunks)
        prompt = (
            f"{SYSTEM_PROMPT}\n\n"
            f"Question:\n{query}\n\n"
            f"Context:\n{context}\n\n"
            "Answer:"
        )
        payload = {
            "model": self.settings.llm_ollama_model,
            "prompt": prompt,
            "stream": False,
        }
        response = requests.post(url, json=payload, timeout=120)
        response.raise_for_status()
        body = response.json()
        if "response" in body:
            return body["response"].strip()
        return body.get("message", {}).get("content", "").strip()

    @staticmethod
    def _fallback_answer(query: str, chunks: List[RetrievedChunk], *, error: str | None = None) -> str:
        context_preview = "\n\n".join(f"- {chunk.text[:300]}" for chunk in chunks)
        error_note = f"\n\n[LLM error: {error}]" if error else ""
        return (
            "LLM provider not configured; returning top retrieved snippets instead."
            f"{error_note}\n\nQuestion: {query}\n\nSnippets:\n{context_preview}"
        )
