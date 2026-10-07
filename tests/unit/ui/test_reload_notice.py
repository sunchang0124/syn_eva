import os
import types

import pytest

from syneva.ui import reload_notice


@pytest.fixture
def package(tmp_path, monkeypatch):
    """A fake package root with two loaded modules, and reload mode switched on."""
    monkeypatch.setenv(reload_notice.ENV_VAR, "1")
    root = tmp_path / "syneva"
    (root / "core").mkdir(parents=True)
    files = {"syneva.core.a": root / "core" / "a.py", "syneva.b": root / "b.py"}
    modules = {}
    for name, path in files.items():
        path.write_text("x = 1\n", encoding="utf-8")
        modules[name] = types.SimpleNamespace(__file__=str(path))
    modules["json"] = types.SimpleNamespace(__file__=str(tmp_path / "elsewhere" / "json.py"))
    return types.SimpleNamespace(root=root, modules=modules, state={}, files=files)


def _begin(pkg):
    return reload_notice.begin(root=pkg.root, modules=pkg.modules, state=pkg.state)


def _touch(path, seconds_later=5):
    stat = path.stat()
    os.utime(path, (stat.st_atime, stat.st_mtime + seconds_later))


def test_first_run_is_silent(package, capsys):
    assert _begin(package) is None
    assert capsys.readouterr().out == ""


def test_rerun_without_changes_is_silent(package, capsys):
    _begin(package)
    assert _begin(package) is None  # e.g. a widget click
    assert capsys.readouterr().out == ""


def test_changed_module_reports_reloading_and_reloaded(package, capsys):
    _begin(package)
    _touch(package.files["syneva.core.a"])

    token = _begin(package)
    assert token is not None
    reload_notice.end(token)

    out = capsys.readouterr().out.splitlines()
    assert len(out) == 2
    assert "core/a.py changed" in out[0]
    assert "reloading" in out[0]
    assert "reloaded" in out[1]


def test_change_is_reported_once_across_sessions(package, capsys):
    _begin(package)
    _touch(package.files["syneva.b"])
    assert _begin(package) is not None  # first browser tab picks it up
    assert _begin(package) is None  # a second tab reruns on the same change
    assert capsys.readouterr().out.count("changed") == 1


def test_disabled_without_reload_flag(package, monkeypatch, capsys):
    monkeypatch.delenv(reload_notice.ENV_VAR)
    _begin(package)
    _touch(package.files["syneva.b"])
    token = _begin(package)
    reload_notice.end(token)
    assert token is None
    assert capsys.readouterr().out == ""
