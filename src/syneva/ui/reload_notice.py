"""Console messages for `syn-eva ui --reload`: when a code reload starts and finishes.

Streamlit reruns the app silently after a source change, so without these
messages the only sign of a reload is the changed behaviour in the browser.
A failed reload needs no message here: Streamlit prints the traceback itself.
"""

from __future__ import annotations

import os
import sys
import time
import types
from collections.abc import Mapping, MutableMapping
from pathlib import Path
from typing import Any

ENV_VAR = "SYNEVA_UI_RELOAD"

# Kept in a module with no __file__ so Streamlit never watches or evicts it,
# which lets the snapshot survive the reload of every syneva module.
_STATE_MODULE = "_syneva_ui_reload_state"

_PACKAGE_ROOT = Path(__file__).resolve().parent.parent


def _shared_state() -> MutableMapping[str, Any]:
    module = sys.modules.get(_STATE_MODULE)
    if module is None:
        module = sys.modules[_STATE_MODULE] = types.ModuleType(_STATE_MODULE)
    return module.__dict__


def _snapshot(root: Path, modules: Mapping[str, Any]) -> dict[str, float]:
    """Modification time of every loaded module file under ``root``.

    Loaded modules are the ones Streamlit watches, so a change to any other
    file (e.g. the CLI) does not trigger a rerun and is not reported either.
    """
    snapshot: dict[str, float] = {}
    for module in list(modules.values()):
        file = getattr(module, "__file__", None)
        if not file:
            continue
        path = Path(file).resolve()
        if root not in path.parents:
            continue
        try:
            snapshot[path.relative_to(root).as_posix()] = path.stat().st_mtime
        except OSError:  # deleted since it was imported
            continue
    return snapshot


def begin(
    root: Path | None = None,
    modules: Mapping[str, Any] | None = None,
    state: MutableMapping[str, Any] | None = None,
) -> float | None:
    """Call at the start of each app run.

    Prints which files changed when this run follows a code change, and
    returns the time of that change for :func:`end`. Returns None otherwise.
    """
    if os.environ.get(ENV_VAR) != "1":
        return None
    state = _shared_state() if state is None else state
    current = _snapshot(root or _PACKAGE_ROOT, sys.modules if modules is None else modules)
    previous = state.get("snapshot")
    state["snapshot"] = current
    # Nothing to report on the first run, or on a rerun from a widget or a second tab.
    if previous is None or previous == current:
        return None
    changed = sorted(
        f for f in current.keys() | previous.keys() if current.get(f) != previous.get(f)
    )
    print(f"[syneva] {', '.join(changed)} changed, reloading the UI...", flush=True)
    return max((current[f] for f in changed if f in current), default=time.time())


def end(changed_at: float | None) -> None:
    """Call at the end of a successful app run, with the value from :func:`begin`."""
    if changed_at is None:
        return
    print(f"[syneva] UI reloaded ({time.time() - changed_at:.1f} s after the change)", flush=True)
