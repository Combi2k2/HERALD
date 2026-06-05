"""Tests for dataset path layout."""

from datetime import datetime

from herald.data import RunPaths
from herald.data.paths import PHASE1_ROOT, RAW_ROOT, RUN_ID, RUN_ID_FORMAT, RUN_ROOT


def test_run_layout():
    assert len(RUN_ID) == 13
    assert RUN_ID[6] == "_"
    assert datetime.strptime(RUN_ID, RUN_ID_FORMAT)
    assert RUN_ROOT == RUN_ROOT.parent / RUN_ID
    assert RAW_ROOT == RUN_ROOT / "raw"
    assert PHASE1_ROOT == RUN_ROOT / "phase1"


def test_run_paths():
    paths = RunPaths(run_id="010626_135959")
    assert paths.root == paths.root.parent / "010626_135959"
    assert paths.raw == paths.root / "raw"
    assert paths.phase1 == paths.root / "phase1"
    assert paths.map_overlay.parent == paths.raw
    assert paths.scene_graph.parent == paths.phase1
