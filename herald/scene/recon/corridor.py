"""Spatial-temporal corridor filter for in-video dynamic objects (CausalNav-style).

A Boxer track already *is* a spatial-temporal corridor: the per-frame world-frame
centroids of the same object, associated by Hungarian matching. We separate real
motion from a static object whose detection box merely *slides* (OWL firing on
overlapping windows of a long bench, drifting the box as the camera pans) by the
per-step motion, NOT the cumulative extent: a slide takes tiny steps that only add
up over many frames, while a mover displaces every frame. `motion` = the median
frame-to-frame centroid step (median rejects lift jitter and the occasional
window-jump). Cumulative-span metrics can't tell a slow slide from real motion."""

from __future__ import annotations

import numpy as np

MIN_TRACK = 3  # a corridor needs at least this many observations to be judged


def motion(traj) -> float:
    """Robust per-step motion: the median frame-to-frame displacement of a track's
    world centroids. Small for a static object (even one whose box slowly slides --
    each step is tiny); large for a genuine mover (it displaces every frame)."""
    a = np.asarray(traj, np.float32).reshape(-1, 3)
    if len(a) < MIN_TRACK:
        return 0.0
    return float(np.median(np.linalg.norm(np.diff(a, axis=0), axis=1)))


def classify_static(objects: list, *, max_step: float = 0.2, min_support: int = 0):
    """Split objects into (static, dynamic) by per-step motion, annotating o["motion"].

    `objects` are Boxer dicts carrying a `traj` of per-frame world centroids (M,3). An
    object is dynamic only if its per-step motion exceeds `max_step` AND it has at
    least `min_support` observations -- a real mover displaces consistently and is
    seen enough, so this avoids dropping a static object off a noisy few-frame track.
    Movers are returned separately so they can be removed (and inspected/rendered)."""
    static, dynamic = [], []
    for o in objects:
        o["motion"] = motion(o.get("traj", ()))
        is_dyn = o["motion"] > max_step and o.get("support", 0) >= min_support
        (dynamic if is_dyn else static).append(o)
    return static, dynamic
