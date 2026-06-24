# syneva — Phase 0c-c: evaluation presets — Design

**Date:** 2026-06-24
**Status:** Approved (pending spec review)
**Depends on:** syneva v0.1 + 0a + 0b + 0c-a (Gower) + 0c-b (holdout). Public API:
`evaluate`/`evaluate_with`, `registry.select`, `Metadata`, `SynevaError`, CLI `evaluate`, UI `run_report`/`_sidebar`.
**Roadmap:** component C3 (methodology rigor), sub-spec C3c. Sibling remaining: multi-classifier utility.

## Goal

Add named evaluation **presets** (`fast`, `full`, `privacy`) so users — and the UI/CLI — can pick a
curated selection profile instead of hand-wiring the five selection knobs. A preset only supplies
*defaults*; any explicitly-passed knob overrides it. With `preset=None` and no explicit knobs, behavior
is byte-identical to today (behavior-preserving; golden snapshot stays green).

## Background — the five selection knobs

Metric selection in `evaluate()`/`evaluate_with()` is driven by:
- `tiers: Sequence[str]` — default `("core",)`; subset of {core, extended, custom}.
- `cs: Sequence[str] | None` — default `None` (= all Cs); restricts to given C dimensions.
- `run_utility: bool` — default `False`; runs the model-training utility (TSTR) metrics.
- `run_fairness: bool` — default `False`; runs the fairness metrics.
- `distance: str` — default `"euclidean"`; `"gower"` is the O(n²) mixed-type alternative.

A preset is a named bundle of these five values. (Decision: pure knob bundle — no per-metric `cost`
tagging. If a single metric later proves too slow, a `cost` field can be added then. YAGNI.)

## Component: `src/syneva/core/presets.py`

```python
from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class Preset:
    tiers: tuple[str, ...]
    cs: tuple[str, ...] | None       # None = all Cs
    run_utility: bool
    run_fairness: bool
    distance: str

PRESETS: dict[str, Preset] = {
    "fast": Preset(
        tiers=("core",), cs=None,
        run_utility=False, run_fairness=False, distance="euclidean",
    ),
    "full": Preset(
        tiers=("core", "extended"), cs=None,
        run_utility=True, run_fairness=True, distance="euclidean",
    ),
    "privacy": Preset(
        tiers=("core", "extended"), cs=("compliance",),
        run_utility=False, run_fairness=False, distance="euclidean",
    ),
}
```

A helper resolves a name to a `Preset` or raises:
```python
def get_preset(name: str) -> Preset:
    if name not in PRESETS:
        raise SynevaError(f"unknown preset '{name}'; valid: {sorted(PRESETS)}")
    return PRESETS[name]
```
(`SynevaError` from `syneva.core.errors`.)

### Preset semantics

| Preset | tiers | cs | run_utility | run_fairness | distance |
|---|---|---|---|---|---|
| `fast` | `("core",)` | all | False | False | euclidean |
| `full` | `("core","extended")` | all | True | True | euclidean |
| `privacy` | `("core","extended")` | `("compliance",)` | False | False | euclidean |

- **fast** — quick smoke check; core tier only, skips the expensive utility (model training) and
  fairness passes.
- **full** — comprehensive: core + extended across all five Cs plus utility and fairness. `distance`
  stays euclidean — gower is O(n²) and remains an explicit opt-in even here.
- **privacy** — disclosure-risk focus: only the Compliance dimension (DCR, NNDR, MIA, k-anonymity,
  ε-identifiability, attribute-disclosure). Pair with a `holdout` for the relative privacy baselines.
  **Accepted limitation:** the pure-bundle approach cannot add coverage's `authenticity` to `privacy`
  without pulling in *all* coverage metrics, so authenticity stays out of `privacy` (reachable via
  `full`). Documented, not a bug.

## Resolution & override (explicit wins)

The five preset-controlled params switch to an internal sentinel so "explicitly passed" is detectable:

```python
_UNSET = object()
```

`evaluate` / `evaluate_with` signatures change those five params' defaults to `_UNSET` and add
`preset: str | None = None` (placed before `random_state`). Resolution, before `reg.select(...)`:

```python
preset_obj = get_preset(preset) if preset is not None else None

def _resolve(value, attr, hard_default):
    if value is not _UNSET:
        return value                       # explicit arg wins
    if preset_obj is not None:
        return getattr(preset_obj, attr)   # else preset value
    return hard_default                    # else today's default

tiers = _resolve(tiers, "tiers", ("core",))
cs = _resolve(cs, "cs", None)
run_utility = _resolve(run_utility, "run_utility", False)
run_fairness = _resolve(run_fairness, "run_fairness", False)
distance = _resolve(distance, "distance", "euclidean")
```

After resolution the existing distance validation and `reg.select(...)` run exactly as today on the
resolved values. `evaluate` forwards `preset=preset` AND the five (possibly `_UNSET`) values down to
`evaluate_with`; resolution happens once, in `evaluate_with` (the single authority). To avoid
double-resolution, `evaluate` passes the raw params straight through (still `_UNSET` if untouched).

**Behavior preservation:** with `preset=None` and no explicit knobs, every knob resolves to its
hard default — identical to today. The golden snapshot (no preset) must stay green.

## CLI

`src/syneva/cli/main.py` `evaluate` command gains:
```python
preset: str | None = typer.Option(None, help="evaluation preset: fast | full | privacy"),
```

`_UNSET` is imported from `syneva.core.presets`. The CLI maps each preset-controlled option to
`_UNSET` when the user did not provide it, so the runner's explicit-wins resolution sees "not given":

- `--tiers`: Typer default changes from `"core"` to `None`. If `None` → forward `tiers=_UNSET`; if
  given → forward `tuple(t.strip() ...)` as today.
- `--cs`: already defaults `None`. If `None` → forward `cs=_UNSET`; if given → split as today.
  (None-as-"all" is now expressed only via a preset or the runner's hard default.)
- `--run-utility`: a `bool` flag cannot express "unset", so the rule is simple: forward
  `run_utility=True` when the flag is set, else `run_utility=_UNSET`. Thus `--preset full` enables
  utility, a bare run keeps today's `False`, and `--run-utility` forces it on over any preset.

`run_fairness` and `distance` are not currently CLI options and stay preset/runner-controlled (forward
nothing → they resolve via preset or hard default). An unknown preset raises `SynevaError`, already
caught by the command's `except SynevaError` block (exit code 2).

## UI

`src/syneva/ui/core.py` `run_report` gains `preset: str | None = None` (before `random_state`),
forwarded to `evaluate_with`. `src/syneva/ui/app.py` `_sidebar()` adds a "Profile" `st.selectbox` at
the top of the controls:
```python
profile = st.selectbox("Profile", ["Custom", "fast", "full", "privacy"], index=0)
```
- `Custom` → today's behavior: the manual metric/evaluator selection drives `run_report` (preset stays
  `None`).
- a named profile → `run_report(..., preset=profile)`; the manual evaluator multiselect is disabled
  (greyed) and the resolved profile is used. The returned config dict carries `"preset"`
  (None for Custom). `main()` passes `preset=cfg["preset"]`.

## Error handling

- Unknown preset name → `SynevaError("unknown preset '<x>'; valid: ['fast', 'full', 'privacy']")`.
- Conflicts never error: an explicit knob always overrides the preset (explicit-wins).

## Testing

- **Resolution:** `evaluate(..., preset="fast")` selects only core-tier metrics and runs neither
  utility nor fairness; `preset="full"` includes extended-tier metrics and runs utility; `preset=
  "privacy"` yields only compliance-C metrics.
- **Override:** `evaluate(..., preset="fast", run_utility=True)` runs utility despite the preset;
  `preset="privacy", cs=None` widens back to all Cs.
- **Behavior preservation:** `evaluate(real, syn)` with no preset selects the identical metric set as
  before this change; the golden snapshot stays green (NEVER `--update-golden`).
- **Unknown preset:** `evaluate(..., preset="nope")` raises `SynevaError` matching "unknown preset".
- **`get_preset`:** returns the right `Preset` for each name; raises for an unknown name.
- **CLI:** `syneva evaluate --preset full ...` writes a scorecard whose results include an
  extended-tier metric; unknown preset exits non-zero.
- **UI:** `run_report(real, syn, meta, [], preset="fast")` runs and returns core metrics; the sidebar
  selectbox wiring passes `preset` through (`test_run_report_accepts_preset`).
- 85% coverage gate holds.

## Out of scope
Multi-classifier utility (separate sub-spec). Per-metric `cost` tagging (deferred until a metric
demonstrably needs excluding). User-defined custom presets / preset files (YAGNI for now).

## Open questions
None outstanding.
