"""Tests for progress helpers."""

from herald.scene.common.progress import iter_progress, progress_task


def test_iter_progress_disabled():
    items = list(iter_progress(range(5), desc="test", disable=True))
    assert items == [0, 1, 2, 3, 4]


def test_progress_task_disabled():
    with progress_task("test", total=3, disable=True) as bar:
        bar.update(3)
        bar.set_postfix_str("done")
