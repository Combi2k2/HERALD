"""Tests for progress helpers."""

from herald.scene.common.progress import iter_progress


def test_iter_progress_disabled():
    items = list(iter_progress(range(5), desc="test", disable=True))
    assert items == [0, 1, 2, 3, 4]
