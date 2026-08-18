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


class TextEncoder:
    """Dedicated text-embedding model (sentence-transformers style: mean-pooled token embeddings,
    L2-normalized) via plain `transformers` -- no sentence-transformers dependency. This is the
    default encoder for object label embeddings: a text-only model gives crisper synonym/related
    geometry than a vision-language text tower. Lazy-loaded on first `encode`; every distinct
    string is cached, so re-encoding the small object vocabulary is free.

    Default `all-MiniLM-L6-v2` (384-d) needs no query/passage prefix; swap `model_id` freely."""

    def __init__(self, model_id: str = "sentence-transformers/all-MiniLM-L6-v2",
                 device: str | None = None) -> None:
        self.model_id = model_id
        self._device = device
        self._model = None
        self._tok = None
        self._dim: int | None = None
        self._cache: dict[str, np.ndarray] = {}

    def _lazy(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModel, AutoTokenizer
        if self._device is None:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._tok = AutoTokenizer.from_pretrained(self.model_id)
        self._model = AutoModel.from_pretrained(self.model_id).to(self._device).eval()
        self._dim = int(self._model.config.hidden_size)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        texts = list(texts)
        missing = [t for t in texts if t not in self._cache]
        if missing:
            self._lazy()
            import torch
            with torch.no_grad():
                inp = self._tok(missing, padding=True, truncation=True,
                                return_tensors="pt").to(self._device)
                out = self._model(**inp).last_hidden_state            # (B, T, D)
                mask = inp["attention_mask"].unsqueeze(-1).to(out.dtype)   # (B, T, 1)
                pooled = (out * mask).sum(1) / mask.sum(1).clamp(min=1e-9)  # mean over real tokens
                pooled = torch.nn.functional.normalize(pooled, dim=-1)
                feats = pooled.cpu().numpy().astype(np.float32)
            for t, v in zip(missing, feats):
                self._cache[t] = v
        dim = self._dim or 0
        return np.stack([self._cache[t] for t in texts]) if texts else np.zeros((0, dim), np.float32)


class SiglipTextEncoder:
    """Optional text encoder using SigLIP's text tower. Vectors are L2-normalized and live in the
    SAME space as SigLIP image features, so text-label embeddings stay comparable to any future
    crop embeddings -- use this only when you want text<->image alignment; otherwise prefer the
    text-only `TextEncoder`. The model is lazy-loaded on first `encode`, and every distinct string
    is cached -- so re-encoding the small object vocabulary is free."""

    def __init__(self, model_id: str = "google/siglip-base-patch16-224", device: str | None = None) -> None:
        self.model_id = model_id
        self._device = device
        self._model = None
        self._proc = None
        self._cache: dict[str, np.ndarray] = {}

    def _lazy(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModel, AutoProcessor
        if self._device is None:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = AutoModel.from_pretrained(self.model_id).to(self._device).eval()
        self._proc = AutoProcessor.from_pretrained(self.model_id)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        texts = list(texts)
        missing = [t for t in texts if t not in self._cache]
        if missing:
            self._lazy()
            import torch
            with torch.no_grad():
                inp = self._proc(text=missing, padding="max_length", truncation=True,
                                 return_tensors="pt").to(self._device)
                feats = self._model.get_text_features(**inp)
                feats = torch.nn.functional.normalize(feats, dim=-1).cpu().numpy().astype(np.float32)
            for t, v in zip(missing, feats):
                self._cache[t] = v
        return np.stack([self._cache[t] for t in texts]) if texts else np.zeros((0, self.dim), np.float32)
