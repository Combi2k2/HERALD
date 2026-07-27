"""Minimal RGB-stream reconstruction: VGGT-Omega geometry + SAM2 masks -> semantic cloud.

The three submodules are independent of each other; the end-to-end script
(scripts/test_recon.py, via pipeline.ReconStream) wires them. Data flows as plain numpy arrays:
vggt.VggtStream gives {"K","c2w","depth","conf"} per frame, sam2.Sam2Segmenter
gives per-frame instance masks, and fusion.Fuser associates the masks into
objects (by appearance + occupancy) and votes them into a labeled cloud.

vggt is not re-exported here (importing it pulls in torch + vggt_omega), and
Sam2Segmenter / SiglipEmbedder are re-exported lazily (they pull in torch +
transformers), so fusion-only users pay for neither. Import vggt directly as
herald.scene.recon.vggt.
"""

from herald.scene.recon.fusion import Fuser
from herald.scene.recon.utils import (
    bbox_crop,
    erode_labels,
    filter_clusters,
    load_cloud,
    save_cloud,
    unproject_labeled,
    write_ply,
)

__all__ = [
    "Fuser",
    "Sam2Segmenter",
    "Sam2Tracker",
    "SiglipEmbedder",
    "TrackReconStream",
    "VoteCloud",
    "bbox_crop",
    "erode_labels",
    "filter_clusters",
    "load_cloud",
    "save_cloud",
    "unproject_labeled",
    "write_ply",
]

def __getattr__(name: str):
    if name == "Sam2Segmenter":
        from herald.scene.recon.sam2 import Sam2Segmenter

        return Sam2Segmenter
    if name == "SiglipEmbedder":
        from herald.scene.recon.embed import SiglipEmbedder

        return SiglipEmbedder
    if name in ("Sam2Tracker", "VoteCloud", "TrackReconStream"):
        from herald.scene.recon import track

        return getattr(track, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
