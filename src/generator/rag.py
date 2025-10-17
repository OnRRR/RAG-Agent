from __future__ import annotations

from dataclasses import dataclass

from src.config import Settings
from src.generator.llm import LLMGenerator
from src.generator.simple import EchoGenerator
from src.generator.types import GeneratedAnswer, GeneratorProtocol
from src.index.embedding import SentenceTransformerEmbedder
from src.retriever.faiss_retriever import FaissRetriever


@dataclass(slots=True)
class RAGPipeline:
    retriever: FaissRetriever
    generator: GeneratorProtocol

    @classmethod
    def from_settings(cls, settings: Settings) -> "RAGPipeline":
        embedder = SentenceTransformerEmbedder(model_name=settings.embedding_model)
        retriever = FaissRetriever(
            index_dir=settings.index_dir,
            embedder=embedder,
            top_k=settings.retriever_top_k,
        )
        generator = cls._build_generator(settings)
        return cls(retriever=retriever, generator=generator)

    @staticmethod
    def _build_generator(settings: Settings) -> GeneratorProtocol:
        if settings.llm_provider in {"openai", "ollama"}:
            return LLMGenerator(settings=settings)
        return EchoGenerator()

    def answer(self, query: str, *, top_k: int | None = None) -> GeneratedAnswer:
        chunks = self.retriever.retrieve(query, top_k=top_k)
        return self.generator.generate(query, chunks)

    def set_generator(self, generator: GeneratorProtocol) -> None:
        self.generator = generator

    def set_retriever(self, retriever: FaissRetriever) -> None:
        self.retriever = retriever
