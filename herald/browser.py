"""Browser launch helpers."""

from __future__ import annotations

import os
import webbrowser


def suppress_gtk_atk_bridge_warning() -> None:
    """Drop obsolete ``atk-bridge`` from GTK_MODULES (GTK 3+ provides this natively)."""
    raw = os.environ.get("GTK_MODULES", "")
    parts = [
        part
        for chunk in raw.split(":")
        for part in chunk.split()
        if part and part != "atk-bridge"
    ]
    os.environ["GTK_MODULES"] = ":".join(parts)


def open_browser(url: str) -> None:
    """Open ``url`` in the default browser."""
    suppress_gtk_atk_bridge_warning()
    webbrowser.open(url)
