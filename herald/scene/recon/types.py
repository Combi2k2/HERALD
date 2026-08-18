"""The unified scene data structure, shared by Tier-1 recon and Tier-2 merge.

A per-video reconstruction (a "session") and a merged multi-session persistent map are the
SAME structure -- `SceneMap` (points + colours + `SceneObject`s). Tier-2 consumes a
`SceneMap`, never raw frames; `SessionResult` is an alias used at the Tier-1 output boundary.
Poses/points are in the TartanGround NED world frame (same frame Boxer objects come back in).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class SceneObject:
    """One object in a scene map -- the single record used by both per-session recon and the
    merged persistent map. Geometry is an oriented box (OBB) in the NED world frame."""

    # --- identity: stable within a map, preserved across merges. NOT the per-video track id
    #     (that lives in `sessions` as provenance, since it is unique only within one session)
    uid: int

    # --- 3D OBB ---
    center: np.ndarray           # (3,)
    half_size: np.ndarray        # (3,) half-extents
    quat_xyzw: np.ndarray        # (4,)

    # --- semantics ---
    label: str                   # top-voted label
    labels: dict                 # label -> vote count (fused across observations)

    # --- evidence ---
    conf: float = 0.0            # fused detection confidence, (score2d + score3d)/2 in [0,1]
    support: int = 0             # total observations (frames) across all contributing sessions

    # --- provenance (per contributing session) ---
    #     session_id -> {"support": int, "conf": float, "track_id": int,
    #                    "crops": [(frame_id, box_xyxy), ...]}
    #     grows as sessions are merged in -> the evidence a Tier-2 retire will reason over.
    #     The `crops` are lightweight render-only references (frame index + 2D box), NOT
    #     pixels -- resolved against `SceneMap.meta["sources"][session_id]` (a path) at
    #     render time; nothing in the processing pipeline reads them.
    sessions: dict = field(default_factory=dict)

    # --- features ---
    embedding: np.ndarray | None = None        # (D,) visual-semantic embedding; None until computed


@dataclass
class SceneMap:
    """A scene: coloured point cloud + object OBBs. One video's recon and a merged
    multi-session map are the same type -- the firewall Tier-2 consumes. `dynamic` keeps the
    filtered-out movers (for viz/analysis, not dropped). `meta` carries map-level bookkeeping
    such as the contributing session ids."""

    points: np.ndarray           # (N,3) scene cloud
    colors: np.ndarray           # (N,3) uint8
    objects: list                # list[SceneObject]
    dynamic: list = field(default_factory=list)   # filtered-out movers
    meta: dict = field(default_factory=dict)      # {"sessions": [...], "sources": {sid: path}} -- `sources`
                                                  # maps a session id to its frames-folder/video path (crop refs)


# Tier-1 output boundary name (a single session's reconstruction is just a SceneMap).
SessionResult = SceneMap
