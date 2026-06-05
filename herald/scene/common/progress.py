"""CLI progress bars for long-running scene-init steps."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import TypeVar

T = TypeVar("T")


def iter_progress(
    iterable: Iterable[T],
    *,
    desc: str,
    total: int | None = None,
    disable: bool = False,
    unit: str = "it",
    leave: bool = True,
) -> Iterator[T]:
    """Wrap an iterable with ``tqdm`` when progress is enabled."""
    if disable:
        yield from iterable
        return
    from tqdm import tqdm

    yield from tqdm(iterable, desc=desc, total=total, unit=unit, leave=leave)


def progress_task(
    desc: str,
    *,
    total: int,
    disable: bool = False,
    unit: str = "it",
):
    """Context manager for manual progress updates."""
    if disable:
        return _NullProgress()
    from tqdm import tqdm

    return tqdm(total=total, desc=desc, unit=unit)


class _NullProgress:
    def update(self, n: int = 1) -> None:
        return None

    def set_postfix_str(self, s: str, refresh: bool = True) -> None:
        return None

    def close(self) -> None:
        return None

    def __enter__(self):
        return self

    def __exit__(self, *args) -> None:
        return None
