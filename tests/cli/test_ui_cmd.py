import sys

import pytest

from syneva.cli import main as cli


def test_ui_builds_streamlit_command(monkeypatch):
    captured = {}

    def fake_run(cmd, *args, **kwargs):
        captured["cmd"] = cmd

        class _R:
            returncode = 0

        return _R()

    monkeypatch.setattr(cli.subprocess, "run", fake_run)
    with pytest.raises(SystemExit) as exc_info:
        cli.ui()
    assert exc_info.value.code == 0  # propagates the subprocess returncode
    assert captured["cmd"][0] == sys.executable
    assert "streamlit" in captured["cmd"]
    assert "run" in captured["cmd"]
    assert captured["cmd"][-1].endswith("app.py")


def test_ui_missing_streamlit_errors(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "streamlit":
            raise ImportError("no streamlit")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(SystemExit) as exc_info:
        cli.ui()
    assert exc_info.value.code == 2
