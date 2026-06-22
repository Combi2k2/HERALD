"""Text embedding encoders for scene graph nodes."""

from __future__ import annotations

import hashlib
from typing import Protocol, Sequence

import numpy as np


class Encoder(Protocol):
    """Encode text descriptions into dense vectors."""

    model_id: str

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Return array of shape (len(texts), dim)."""
        ...


class StubEncoder:
    """Deterministic pseudo-embeddings for testing without ML deps."""

    model_id = "stub-v0"

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            seed = int.from_bytes(digest[:8], "big")
            rng = np.random.default_rng(seed)
            vec = rng.standard_normal(self.dim).astype(np.float32)
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec /= norm
            vectors[i] = vec
        return vectors
