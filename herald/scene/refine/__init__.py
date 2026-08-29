"""Tier-2 refine: reconcile many Tier-1 SessionResults into one persistent map.

`align` registers a session cloud onto a canonical/other-session cloud by energy-based Sim3
(RANSAC ground-level -> scale/yaw/translation). `reconcile` then merges objects via a
two-pass gate + union-find (update/insert), carrying per-session provenance. Evidence-gated
retire is not built yet.
"""

from __future__ import annotations

from herald.scene.refine.align import AlignResult, align, align_sessions, fit_ground_plane
from herald.scene.refine.reconcile import (
    load_meta, load_session, overlap_merge, reconcile, save_session, transform_object,
)

__all__ = ["AlignResult", "align", "align_sessions", "fit_ground_plane", "reconcile",
           "overlap_merge", "transform_object", "save_session", "load_session", "load_meta"]
