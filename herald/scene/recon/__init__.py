"""Minimal RGB-stream reconstruction: VGGT-Omega geometry + SAM2 masks -> semantic cloud.

The three submodules are independent of each other; the end-to-end script
(scripts/recon_stream.py) wires them. Data flows as plain numpy arrays:
vggt.VggtStream gives {"K","c2w","depth","conf"} per frame, sam2.Sam2Stream
gives per-frame label maps, and fusion.Fuser votes both into a labeled cloud.
Both streams expose run_chunk / run_video / run_stream plus the incremental
push/finish interface.

vggt is not re-exported here (importing it pulls in torch + vggt_omega) and
Sam2Stream is re-exported lazily (sam2 pulls in torch + transformers), so
fusion-only users pay for neither. Import vggt directly as
herald.scene.recon.vggt.
"""

from herald.scene.recon.fusion import Fuser
from herald.scene.recon.utils import (
    erode_labels,
    filter_clusters,
    load_cloud,
    save_cloud,
    unproject_labeled,
    write_ply,
)

__all__ = [
    "Fuser",
    "Sam2Stream",
    "erode_labels",
    "filter_clusters",
    "load_cloud",
    "save_cloud",
    "unproject_labeled",
    "write_ply",
]

def __getattr__(name: str):
    if name == "Sam2Stream":
        from herald.scene.recon.sam2 import Sam2Stream

        return Sam2Stream
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
