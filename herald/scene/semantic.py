"""Semantic embeddings for scene objects.

Text labels are kept for rendering/inspection; this module fills the separate `embedding`
vector used for computational semantic reasoning. An object's embedding is the **vote-weighted
mean of its label embeddings** -- i.e. the EMA of the per-frame text embeddings collapsed to
closed form (each frame contributes its label's vector; a label's text embedding is
deterministic, so the running average reduces to Σ votes_l · e_l / Σ votes_l). Unanimous votes
-> exactly that one label's vector.

The encoder is any `services.embeddings.Encoder` -- the default `TextEncoder` (a text-only
sentence-embedding model), or `StubEncoder` for tests. Only distinct labels are encoded, so
re-embedding a whole map is cheap.
"""

from __future__ import annotations

import numpy as np


def embed_objects(objects, encoder, *, normalize: bool = True):
    """Set `o.embedding` for every object from its `labels` vote dict, in place. Objects with
    no label votes are left untouched (embedding stays None). Returns `objects`."""
    if not objects:
        return objects
    vocab = sorted({lab for o in objects for lab in o.labels})
    if not vocab:
        return objects
    vecs = np.asarray(encoder.encode(vocab), np.float32)          # (V, D)
    idx = {lab: i for i, lab in enumerate(vocab)}
    for o in objects:
        if not o.labels:
            continue
        labs = list(o.labels)
        w = np.asarray([o.labels[l] for l in labs], np.float32)   # vote counts
        e = (w[:, None] * vecs[[idx[l] for l in labs]]).sum(0)    # vote-weighted mean
        n = float(np.linalg.norm(e))
        o.embedding = (e / n).astype(np.float32) if (normalize and n > 0) else e.astype(np.float32)
    return objects
