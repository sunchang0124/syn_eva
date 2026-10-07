import os
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

import syneva
from syneva.cli import main as cli
from syneva.ui import reload_notice


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


def _capture_run(monkeypatch):
    captured = {}

    def fake_run(cmd, *args, **kwargs):
        captured["cmd"] = cmd
        captured["env"] = kwargs.get("env")

        class _R:
            returncode = 0

        return _R()

    monkeypatch.setattr(cli.subprocess, "run", fake_run)
    return captured


def test_ui_default_does_not_enable_reload(monkeypatch):
    captured = _capture_run(monkeypatch)
    with pytest.raises(SystemExit):
        cli.ui()
    assert "--server.runOnSave" not in captured["cmd"]
    assert captured["env"] is None  # inherits the parent environment unchanged


def test_ui_reload_reruns_on_save_and_watches_the_package(monkeypatch):
    monkeypatch.setenv("PYTHONPATH", "existing")
    captured = _capture_run(monkeypatch)
    with pytest.raises(SystemExit):
        cli.ui(reload=True)
    cmd = captured["cmd"]
    assert cmd[cmd.index("--server.runOnSave") + 1] == "true"
    assert cmd[-1].endswith("app.py")
    # Streamlit only watches modules next to app.py or on PYTHONPATH, so the
    # folder containing the syneva package must be on it, ahead of what was there.
    package_root = str(Path(syneva.__file__).resolve().parent.parent)
    assert captured["env"]["PYTHONPATH"].split(os.pathsep) == [package_root, "existing"]
    # Switches on the console messages when a reload starts and finishes.
    assert captured["env"][reload_notice.ENV_VAR] == "1"


def test_ui_reload_flag_on_command_line(monkeypatch):
    captured = _capture_run(monkeypatch)
    result = CliRunner().invoke(cli.app, ["ui", "--reload"])
    assert result.exit_code == 0
    assert "--server.runOnSave" in captured["cmd"]


def test_ui_reload_warns_when_not_an_editable_install(monkeypatch, capsys):
    fake_init = Path("venv", "Lib", "site-packages", "syneva", "__init__.py").resolve()
    monkeypatch.setattr(cli.syneva, "__file__", str(fake_init))
    _capture_run(monkeypatch)
    with pytest.raises(SystemExit):
        cli.ui(reload=True)
    assert "editable install" in capsys.readouterr().err


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
