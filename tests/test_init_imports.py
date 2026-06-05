"""Guard tests for scene init module wiring."""

import importlib
from pathlib import Path


def test_init_module_imports_cleanly():
    pipeline = importlib.import_module("herald.scene.init.pipeline")
    io_mod = importlib.import_module("herald.scene.init.io")
    assert callable(pipeline.build_scene_graph)
    assert callable(io_mod.save_classifications)


def test_init_uses_pathways_filter():
    from herald.scene.init import pipeline as init_mod

    text = Path(init_mod.__file__).read_text(encoding="utf-8")
    assert "from herald.scene.init.pathways import filter_walkable_highways" in text
