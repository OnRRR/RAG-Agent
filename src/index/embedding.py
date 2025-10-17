from __future__ import annotations

from typing import Iterable, List

import numpy as np
from sentence_transformers import SentenceTransformer


class SentenceTransformerEmbedder:
    """Thin wrapper around SentenceTransformer with sensible defaults for CPU usage."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self._model: SentenceTransformer | None = None

    @property
    def dimension(self) -> int:
        model = self._ensure_model()
        return model.get_sentence_embedding_dimension()

    def embed(self, texts: Iterable[str]) -> np.ndarray:
        model = self._ensure_model()
        # SentenceTransformer returns a numpy array when convert_to_numpy=True.
        embeddings = model.encode(list(texts), convert_to_numpy=True, batch_size=64, normalize_embeddings=True)
        return embeddings.astype("float32", copy=False)

    def _ensure_model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self.model_name, device="cpu")
        return self._model

