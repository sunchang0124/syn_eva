# Evaluation Presets (Phase 0c-c) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add named evaluation presets (`fast`/`full`/`privacy`) as pure bundles of the five selection knobs, resolved with explicit-wins override and byte-identical behavior when `preset=None`.

**Architecture:** A new `presets.py` defines a frozen `Preset` dataclass, a `PRESETS` registry, `get_preset()`, and an `_UNSET` sentinel. `evaluate`/`evaluate_with` default the five preset-controlled params to `_UNSET` and gain `preset=`; `evaluate_with` resolves each knob (explicit › preset › hard default) once, before `reg.select(...)`. CLI and UI gain a preset option/dropdown.

**Tech Stack:** Python 3.10+, typer, streamlit, pytest, uv.

**Spec:** `docs/superpowers/specs/2026-06-24-syneva-parity-0c-presets.md`.

---

## CRITICAL environment notes
- Use `uv`. Targeted tests MUST use `--no-cov`. Before committing: `uv run ruff check --fix . && uv run ruff format .`, then `git add -A`, commit; if a hook modifies files and aborts, `git add -A` and re-run.
- **Behavior preservation:** with `preset=None` and no explicit-knob changes, every knob resolves to today's hard default. Existing tests must pass UNCHANGED and the golden snapshot must stay green (NEVER `--update-golden`). If an existing test or the golden changes, STOP and report.

## File structure
- Create: `src/syneva/core/presets.py`
- Modify: `src/syneva/core/runner.py` (signatures → `_UNSET` defaults, `preset=`, resolution)
- Modify: `src/syneva/cli/main.py` (`--preset` + `_UNSET` mapping)
- Modify: `src/syneva/ui/core.py` (`run_report` preset param), `src/syneva/ui/app.py` (Profile selectbox)
- Tests under `tests/unit/core/`, `tests/integration/`, `tests/unit/cli/`, `tests/unit/ui/`.

---

## Task 1: `presets.py` module

**Files:** Create `src/syneva/core/presets.py`; Test `tests/unit/core/test_presets.py`.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/core/test_presets.py
import pytest

from syneva.core.errors import SynevaError
from syneva.core.presets import PRESETS, Preset, get_preset


def test_known_presets_exist():
    assert set(PRESETS) == {"fast", "full", "privacy"}


def test_fast_preset_values():
    p = get_preset("fast")
    assert isinstance(p, Preset)
    assert p.tiers == ("core",)
    assert p.cs is None
    assert p.run_utility is False
    assert p.run_fairness is False
    assert p.distance == "euclidean"


def test_full_preset_values():
    p = get_preset("full")
    assert p.tiers == ("core", "extended")
    assert p.cs is None
    assert p.run_utility is True
    assert p.run_fairness is True


def test_privacy_preset_values():
    p = get_preset("privacy")
    assert p.tiers == ("core", "extended")
    assert p.cs == ("compliance",)
    assert p.run_utility is False


def test_unknown_preset_raises():
    with pytest.raises(SynevaError, match="unknown preset"):
        get_preset("nope")


def test_preset_is_frozen():
    p = get_preset("fast")
    with pytest.raises(Exception):
        p.tiers = ("extended",)  # frozen dataclass -> FrozenInstanceError
```

- [ ] **Step 2: Run, expect failure** — `uv run pytest tests/unit/core/test_presets.py --no-cov -v` (ModuleNotFoundError).

- [ ] **Step 3: Implement** — create `src/syneva/core/presets.py`:

```python
from __future__ import annotations

from dataclasses import dataclass

from syneva.core.errors import SynevaError

# Sentinel marking "argument not supplied" so preset/explicit resolution can tell
# an explicit value from an omitted one (None is a legitimate value for `cs`).
_UNSET = object()


@dataclass(frozen=True)
class Preset:
    tiers: tuple[str, ...]
    cs: tuple[str, ...] | None  # None = all Cs
    run_utility: bool
    run_fairness: bool
    distance: str


PRESETS: dict[str, Preset] = {
    "fast": Preset(
        tiers=("core",),
        cs=None,
        run_utility=False,
        run_fairness=False,
        distance="euclidean",
    ),
    "full": Preset(
        tiers=("core", "extended"),
        cs=None,
        run_utility=True,
        run_fairness=True,
        distance="euclidean",
    ),
    "privacy": Preset(
        tiers=("core", "extended"),
        cs=("compliance",),
        run_utility=False,
        run_fairness=False,
        distance="euclidean",
    ),
}


def get_preset(name: str) -> Preset:
    if name not in PRESETS:
        raise SynevaError(f"unknown preset '{name}'; valid: {sorted(PRESETS)}")
    return PRESETS[name]
```

- [ ] **Step 4: Run, expect pass** — `uv run pytest tests/unit/core/test_presets.py --no-cov -v` (6 passed).

- [ ] **Step 5: Commit** — `git commit -m "feat(core): preset registry (fast/full/privacy) + _UNSET sentinel"`

---

## Task 2: runner preset resolution

**Files:** Modify `src/syneva/core/runner.py`; Test `tests/integration/test_presets.py`.

- [ ] **Step 1: Write the failing test**

```python
# tests/integration/test_presets.py
import pytest

import syneva
from syneva.core.errors import SynevaError


def _names(rep):
    return {r.spec.name for r in rep.results}


def test_fast_preset_core_only_no_utility(real_df, syn_good_df, metadata):
    rep = syneva.evaluate(real_df, syn_good_df, metadata, preset="fast")
    names = _names(rep)
    # core-tier metrics present, no utility metric ran
    assert "tstr_suite" not in names
    assert all(r.spec.tier == "core" for r in rep.results)


def test_full_preset_includes_extended_and_utility(real_df, syn_good_df, metadata):
    rep = syneva.evaluate(real_df, syn_good_df, metadata, preset="full")
    tiers = {r.spec.tier for r in rep.results}
    assert "extended" in tiers
    assert "tstr_suite" in _names(rep)  # utility ran


def test_privacy_preset_compliance_only(real_df, syn_good_df, metadata):
    rep = syneva.evaluate(real_df, syn_good_df, metadata, preset="privacy")
    assert {r.spec.c for r in rep.results} == {"compliance"}


def test_explicit_overrides_preset(real_df, syn_good_df, metadata):
    # fast preset says run_utility=False, but explicit True must win
    rep = syneva.evaluate(
        real_df, syn_good_df, metadata, preset="fast", run_utility=True,
        utility_tasks=[],
    )
    # utility metric now appears despite the fast preset
    assert "tstr_suite" in _names(rep)


def test_no_preset_matches_default_selection(real_df, syn_good_df, metadata):
    a = syneva.evaluate(real_df, syn_good_df, metadata)
    b = syneva.evaluate(real_df, syn_good_df, metadata, preset=None)
    assert _names(a) == _names(b)


def test_unknown_preset_raises(real_df, syn_good_df, metadata):
    with pytest.raises(SynevaError, match="unknown preset"):
        syneva.evaluate(real_df, syn_good_df, metadata, preset="nope")
```

(Confirm fixture names `real_df`/`syn_good_df`/`metadata` from `tests/conftest.py`. `tstr_suite` is the utility metric name; `high_income` exists for auto-suggested tasks — `run_utility=True` with `utility_tasks=[]` exercises the empty-tasks path; if that produces no `tstr_suite` result, instead pass `utility_tasks=[UtilityTask(target="high_income", task_type="classification")]` importing `from syneva.utility.task import UtilityTask`.)

- [ ] **Step 2: Run, expect failure** — `uv run pytest tests/integration/test_presets.py --no-cov -v` (unexpected `preset`).

- [ ] **Step 3: Implement** — edit `src/syneva/core/runner.py`:

(a) Import the sentinel and helper near the top imports:
```python
from syneva.core.presets import _UNSET, get_preset
```

(b) Change `evaluate`'s signature so the five preset-controlled params default to `_UNSET` and add `preset` (keep all other params as-is). The new signature:
```python
def evaluate(
    real: pd.DataFrame | None,
    synthetic: pd.DataFrame,
    metadata: Metadata | None = None,
    *,
    tiers: Sequence[str] | object = _UNSET,
    data_type: str = "static",
    cs: Sequence[str] | None | object = _UNSET,
    utility_tasks: list[UtilityTask] | None = None,
    run_utility: bool | object = _UNSET,
    fairness_specs: list[FairnessSpec] | None = None,
    run_fairness: bool | object = _UNSET,
    distance: str | object = _UNSET,
    holdout: pd.DataFrame | None = None,
    preset: str | None = None,
    random_state: int = 42,
    nan_policy: Literal["drop", "explicit_na", "raise"] = "drop",
) -> Report:
```
Its body forwards the params UNCHANGED (still possibly `_UNSET`) plus `preset=preset` to `evaluate_with`:
```python
    return evaluate_with(
        _default_registry,
        real=real,
        synthetic=synthetic,
        metadata=metadata,
        tiers=tiers,
        data_type=data_type,
        cs=cs,
        utility_tasks=utility_tasks,
        run_utility=run_utility,
        fairness_specs=fairness_specs,
        run_fairness=run_fairness,
        distance=distance,
        holdout=holdout,
        preset=preset,
        random_state=random_state,
        nan_policy=nan_policy,
    )
```

(c) Change `evaluate_with`'s signature identically (the five params → `_UNSET`, add `preset: str | None = None` before `random_state`):
```python
def evaluate_with(
    reg: MetricRegistry,
    *,
    real: pd.DataFrame | None,
    synthetic: pd.DataFrame,
    metadata: Metadata | None = None,
    tiers: Sequence[str] | object = _UNSET,
    data_type: str = "static",
    cs: Sequence[str] | None | object = _UNSET,
    utility_tasks: list[UtilityTask] | None = None,
    run_utility: bool | object = _UNSET,
    fairness_specs: list[FairnessSpec] | None = None,
    run_fairness: bool | object = _UNSET,
    distance: str | object = _UNSET,
    holdout: pd.DataFrame | None = None,
    preset: str | None = None,
    random_state: int = 42,
    nan_policy: str = "drop",
) -> Report:
```

(d) At the very TOP of `evaluate_with`'s body (before the `from syneva.utility.task import suggest_tasks` line is fine, but BEFORE the distance validation), resolve the five knobs:
```python
    preset_obj = get_preset(preset) if preset is not None else None

    def _resolve(value, attr, hard_default):
        if value is not _UNSET:
            return value
        if preset_obj is not None:
            return getattr(preset_obj, attr)
        return hard_default

    tiers = _resolve(tiers, "tiers", ("core",))
    cs = _resolve(cs, "cs", None)
    run_utility = _resolve(run_utility, "run_utility", False)
    run_fairness = _resolve(run_fairness, "run_fairness", False)
    distance = _resolve(distance, "distance", "euclidean")
```
After these five lines, `tiers`/`cs`/`run_utility`/`run_fairness`/`distance` are concrete again and the REST of the function (distance validation, `reg.select(...)`, the utility/fairness pruning, `_instantiate`) is UNCHANGED. Do not touch anything below.

- [ ] **Step 4: Run, expect pass** — `uv run pytest tests/integration/test_presets.py --no-cov -v`.

- [ ] **Step 5: FULL SUITE (behavior-preservation gate)** — `uv run pytest -q`. ALL pass (2 pre-existing skips allowed); golden snapshot UNCHANGED; coverage >= 85%. If the golden fails, STOP and report BLOCKED — a no-preset path changed.

- [ ] **Step 6: Commit** — `git commit -m "feat(runner): preset resolution with explicit-wins override"`

---

## Task 3: CLI `--preset`

**Files:** Modify `src/syneva/cli/main.py`; Test `tests/unit/cli/test_cli.py` (append; confirm the file/path of existing CLI tests first — it may be `tests/unit/cli/test_main.py`).

- [ ] **Step 1: Write the failing test** — append to the existing CLI test module (it uses typer's `CliRunner` and the `app`; reuse those imports and the fixture parquet paths the existing tests use):

```python
def test_cli_preset_full_runs(tmp_path, cli_real, cli_syn):
    # cli_real / cli_syn: paths to fixture parquet files (reuse the pattern existing CLI tests use;
    # if existing tests build paths inline, do the same here instead of fixtures)
    out = tmp_path / "rep"
    result = runner.invoke(
        app,
        ["evaluate", "--real", str(cli_real), "--synthetic", str(cli_syn),
         "--preset", "full", "--out", str(out)],
    )
    assert result.exit_code == 0
    assert (out / "scorecard.json").exists()


def test_cli_unknown_preset_exits_nonzero(tmp_path, cli_syn):
    out = tmp_path / "rep"
    result = runner.invoke(
        app,
        ["evaluate", "--synthetic", str(cli_syn), "--preset", "bogus", "--out", str(out)],
    )
    assert result.exit_code != 0
```

(ADAPT to the existing CLI test style — how it names the `CliRunner` instance (`runner`), how it references fixture files, and the command name. Match reality.)

- [ ] **Step 2: Run, expect failure** — `uv run pytest <cli test path> -k preset --no-cov -v` (no such option `--preset`).

- [ ] **Step 3: Implement** — edit `src/syneva/cli/main.py`:

(a) Import the sentinel:
```python
from syneva.core.presets import _UNSET
```

(b) Add the `--preset` option and change preset-controlled options to map omitted → `_UNSET`. Update the `evaluate` command:
```python
@app.command()
def evaluate(
    real: Path | None = typer.Option(None, exists=True, dir_okay=False),
    synthetic: Path = typer.Option(..., exists=True, dir_okay=False),
    metadata: Path | None = typer.Option(None, exists=True, dir_okay=False),
    out: Path = typer.Option(Path("./syneva-report")),
    tiers: str | None = typer.Option(None, help="comma-separated tiers"),
    cs: str | None = typer.Option(None, help="comma-separated Cs to restrict to"),
    run_utility: bool = typer.Option(False, help="run utility (TSTR) tasks"),
    preset: str | None = typer.Option(None, help="evaluation preset: fast | full | privacy"),
    pdf: bool = typer.Option(False, help="also write scorecard.pdf"),
) -> None:
    out.mkdir(parents=True, exist_ok=True)
    try:
        real_df = _load_df(real) if real else None
        syn_df = _load_df(synthetic)
        meta = _load_metadata(metadata)
        rep = syneva.evaluate(
            real_df,
            syn_df,
            meta,
            tiers=(tuple(t.strip() for t in tiers.split(",")) if tiers else _UNSET),
            cs=(tuple(c.strip() for c in cs.split(",")) if cs else _UNSET),
            run_utility=(True if run_utility else _UNSET),
            preset=preset,
        )
        rep.to_json(out / "scorecard.json")
        rep.to_html(out / "scorecard.html")
        if pdf:
            rep.to_pdf(out / "scorecard.pdf")
        typer.echo(f"wrote scorecard to {out}/")
    except SynevaError as e:
        typer.echo(f"syneva error: {e}", err=True)
        raise typer.Exit(code=2) from e
```
(Note: `--tiers` default changed from `"core"` to `None`; a bare `syneva evaluate` now forwards `tiers=_UNSET` which resolves to the same `("core",)` hard default — behavior unchanged. `--run-utility` forwards `True` when set, else `_UNSET`.)

- [ ] **Step 4: Run, expect pass** — `uv run pytest <cli test path> --no-cov -v` (existing CLI tests UNCHANGED + 2 new pass). If an existing CLI test asserted the old `--tiers core` default behavior and now fails, investigate — a bare run must still produce the core-tier scorecard.

- [ ] **Step 5: Commit** — `git commit -m "feat(cli): --preset option (fast/full/privacy)"`

---

## Task 4: UI Profile dropdown

**Files:** Modify `src/syneva/ui/core.py`, `src/syneva/ui/app.py`; Test append `tests/unit/ui/test_core.py`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/unit/ui/test_core.py
def test_run_report_preset_uses_default_registry():
    real = pd.read_parquet(f"{FIX}/adult_income_real_500.parquet")
    syn = pd.read_parquet(f"{FIX}/adult_income_syn_good_500.parquet")
    # preset drives selection even though selected_names is empty
    rep = core.run_report(real, syn, Metadata.infer(real), [], preset="fast")
    assert len(rep.results) > 0
    assert all(r.spec.tier == "core" for r in rep.results)
```

- [ ] **Step 2: Run, expect failure** — `uv run pytest tests/unit/ui/test_core.py -k preset --no-cov -v` (unexpected `preset`).

- [ ] **Step 3: Implement**

(a) `src/syneva/ui/core.py` — add `preset: str | None = None` to `run_report` (before `random_state`). When a preset is given, run against the DEFAULT full registry with `preset=` (the preset, not the manual selection, drives metrics); otherwise today's selection-registry path. Replace the body:
```python
    if preset is not None:
        from syneva.core.registry import registry as default_registry

        return evaluate_with(
            default_registry,
            real=real,
            synthetic=synthetic,
            metadata=metadata,
            utility_tasks=utility_tasks,
            fairness_specs=fairness_specs,
            holdout=holdout,
            preset=preset,
            random_state=random_state,
        )
    reg = build_selection_registry(selected_names)
    run_utility = any(cls.spec.c == _UTILITY_C for cls in reg.metrics())
    run_fairness = any(cls.spec.c == _FAIRNESS_C for cls in reg.metrics())
    return evaluate_with(
        reg,
        real=real,
        synthetic=synthetic,
        metadata=metadata,
        tiers=("core", "extended", "custom"),
        utility_tasks=utility_tasks,
        run_utility=run_utility,
        fairness_specs=fairness_specs,
        run_fairness=run_fairness,
        holdout=holdout,
        random_state=random_state,
    )
```
(The non-preset branch is the EXISTING body verbatim — keep it byte-identical.)

(b) `src/syneva/ui/app.py` `_sidebar()` — add a Profile selectbox near the top of the controls (e.g. just after the data uploaders / before the evaluator multiselect):
```python
        profile = st.selectbox(
            "Profile",
            ["Custom", "fast", "full", "privacy"],
            index=0,
            help="A preset runs a curated metric set; Custom lets you pick metrics by hand.",
        )
```
When `profile != "Custom"`, disable the manual evaluator multiselect so it's clear the preset is in charge. Find the existing evaluator `st.multiselect(...)` and pass `disabled=profile != "Custom"` to it (add the kwarg; do not otherwise change it). Add to the returned dict:
```python
        "preset": None if profile == "Custom" else profile,
```
(c) `main()` — pass `preset=cfg["preset"]` into the `core.run_report(...)` call (add the kwarg alongside `holdout=cfg["holdout"]`). Also relax the guard at `main()` that blocks running when `not cfg["selected"]`: when a preset is active (`cfg["preset"] is not None`) an empty manual selection is fine. Change the guard so it only errors on empty selection when `cfg["preset"] is None`:
```python
    if cfg and cfg["run"]:
        if cfg["preset"] is None and not cfg["selected"]:
            st.warning("Select at least one evaluator, or choose a Profile.")
        elif cfg["utility_selected"] and not cfg["utility_tasks"]:
            ...
```
(Keep the rest of the `elif` chain exactly as it is. Match the real variable/message style in the file.)

- [ ] **Step 4: Run + boot smoke** — `uv run pytest tests/unit/ui/test_core.py tests/unit/ui/test_app_importable.py --no-cov -v`; then `uv run python -c "import syneva.ui.app"`; headless boot on port 8605 (curl expects 200, log clean), then kill the process.

- [ ] **Step 5: Full suite** — `uv run pytest -q` → all pass, golden UNCHANGED, coverage >= 85%.

- [ ] **Step 6: Commit** — `git commit -m "feat(ui): Profile preset dropdown"`

---

## Self-review notes
- **Spec coverage:** Preset dataclass/registry/get_preset/_UNSET (T1) · runner signatures + resolution + behavior preservation (T2) · CLI --preset + _UNSET mapping (T3) · UI run_report preset path + Profile selectbox + main guard (T4). Behavior preservation gated full-suite in T2 and T4 (golden green; no-preset == today).
- **Type/signature consistency:** `_UNSET` defined once in presets.py, imported by runner + cli; `evaluate` and `evaluate_with` share identical preset-controlled defaults and forward order; `_resolve(value, attr, hard_default)` keys match `Preset` field names (`tiers`/`cs`/`run_utility`/`run_fairness`/`distance`); `run_report(..., preset=None)` non-preset branch is the existing body verbatim.
- **No placeholders:** every step has complete code; CLI/UI test adapt-notes call out the exact reality to confirm before writing.
