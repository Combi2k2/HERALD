"""Tests for GTK browser launch helper."""

import os

from herald.browser import open_browser, suppress_gtk_atk_bridge_warning


def test_suppress_gtk_atk_bridge_warning_removes_module():
    os.environ["GTK_MODULES"] = "atk-bridge:gail:atk-bridge"
    suppress_gtk_atk_bridge_warning()
    assert "atk-bridge" not in os.environ["GTK_MODULES"]
    assert "gail" in os.environ["GTK_MODULES"]


def test_open_browser_calls_webbrowser(monkeypatch):
    opened: list[str] = []
    monkeypatch.setattr("herald.browser.webbrowser.open", lambda url: opened.append(url))
    os.environ["GTK_MODULES"] = "atk-bridge"
    open_browser("http://127.0.0.1:1/")
    assert opened == ["http://127.0.0.1:1/"]
    assert "atk-bridge" not in os.environ["GTK_MODULES"]
