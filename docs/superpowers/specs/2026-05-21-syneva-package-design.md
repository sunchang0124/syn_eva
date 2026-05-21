# Syneva — Python package design (v0.1)

**Status:** approved by Chang Sun, brainstorm 2026-05-21
**Phase:** 1 of 3 (Python package). Phase 2 = web API, Phase 3 = web UI — each gets its own spec.
**Target version:** `syneva` 0.1.0

---

## 1. Scope

A general-purpose Python library for evaluating tabular synthetic data against its real counterpart. Researchers and practitioners install `syneva`, point it at `(real_df, synthetic_df)`, and get back a comprehensive **scorecard** organised around the published **7 Cs framework** (Zamzmi et al., *Communications Engineering* 4:130, 2025; DOI 10.1038/s44172-025-00450-1) plus a parallel **Task-based ML utility** section.

### 1.1 Audience

Open-source — anyone evaluating tabular synthetic data. Not tied to a single disease or institution.

### 1.2 Package identity

| | |
|---|---|
| PyPI / import name | `syneva` |
| Repository folder | `Syn_Eva/` |
| License | Apache-2.0 |
| Python minimum | 3.10 |
| Versioning | Semantic; pre-1.0 may break across minor versions |

### 1.3 Phasing within the package

| Version | Data type supported | Cs implemented |
|---|---|---|
| **v0.1** | static (single table, one row per entity) | Congruence + Coverage + Compliance + Task-based ML utility |
| v0.2 | adds longitudinal (multi-row per entity / sequences) | adds Constraint + Completeness |
| v0.3 | adds relational (linked multi-table) | adds Comprehension + Consistency |

Architecture must support all three data types and all eight evaluation sections from day one. Only the static path and four sections are implemented in v0.1; the rest are declared placeholders that raise informative `NotImplementedError`s.

### 1.4 Three-tier metric ecosystem

| Tier | Source | Selection |
|---|---|---|
| **core** | curated, widely-used, literature-standard metrics | default for `evaluate()` |
| **extended** | metrics from recent (~5-year) synthetic-data literature | opt-in via `tiers=["core", "extended"]` |
| **custom** | user-registered through `@registry.register` | opt-in by tier name |

### 1.5 Outputs available from one `evaluate()` call

`evaluate()` returns an in-memory `Report`; the user explicitly writes any on-disk formats via methods on it. `evaluate()` itself never touches disk.

1. `Report` Python dataclass (always returned)
2. JSON file (machine-readable scorecard, stable schema) — via `report.to_json()`
3. Single-file HTML report (human-readable; embeds figures) — via `report.to_html()`
4. PDF scorecard (rendered from HTML via `weasyprint`) — via `report.to_pdf()`

### 1.6 Visualizations

Every `MetricResult` carries scalar values **and** an optional plot payload. Renderers turn that payload into figures appropriate for the output medium:

- HTML & PDF reports → embedded figures inline
- JSON → figures omitted by default, optional base64 embedding
- Python `Report` → exposes `.plot()` helper per metric for notebook use

**Plot backend:** matplotlib by default (static PNG, used in both HTML and PDF). Pass `interactive=True` to use plotly for HTML interactivity; PDF always uses matplotlib (plotly→PDF is fragile).

### 1.7 Interface surfaces in v0.1

- **Python API:** `syneva.evaluate(real, synthetic, metadata=None, *, tiers=("core",), ...) -> Report`
- **CLI:** `syn-eva evaluate --real real.csv --synthetic syn.csv --out report/`

### 1.8 Explicit non-goals for v0.1

- Web API (Phase 2 — separate spec)
- Web UI (Phase 3 — separate spec)
- Imaging, text, raw time-series, or longitudinal/relational tabular (v0.2/v0.3)
- Data generation (covered by `syn_data_UI`)
- Clinical-specific Delphi / TAM workflows (live in VENI WP3)

---

## 2. Core abstractions

Eight types, one entry point. The contract everything else builds on.

### 2.1 `Metadata`

```python
class ColumnType(StrEnum):
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    DATETIME = "datetime"
    BOOLEAN = "boolean"
    ID = "id"

@dataclass
class ColumnMetadata:
    name: str
    dtype: ColumnType
    sensitive: bool = False                # used by Compliance metrics
    categories: list[str] | None = None
    value_range: tuple[float, float] | None = None

@dataclass
class Metadata:
    columns: dict[str, ColumnMetadata]
    primary_key: str | None = None

    @classmethod
    def infer(cls, df: pd.DataFrame) -> "Metadata": ...
    def override(self, **patches) -> "Metadata": ...
```

`Metadata.infer(df)` is the default path. Power users pass an explicit `Metadata`, or override individual columns. Inference rules: `int64`/`float64` → NUMERIC; `object`/`category` → CATEGORICAL; datetime dtypes → DATETIME; `bool` → BOOLEAN; columns matching `{*_id, id, *_key}` heuristics → ID.

### 2.2 `MetricSpec`

```python
@dataclass(frozen=True)
class MetricSpec:
    name: str                              # stable identifier, e.g. "ks_statistic"
    c: Literal["congruence", "coverage", "compliance", "utility",
               "constraint", "completeness", "comprehension", "consistency"]
    tier: Literal["core", "extended", "custom"]
    data_types: frozenset[Literal["static", "longitudinal", "relational"]]
    requires_real: bool                    # False ⇒ synthetic-only evaluation OK
    scope: Literal["per-column", "pairwise", "table-level"]
```

`requires_real=False` makes synthetic-only evaluation (Constraint / Completeness / Comprehension in later versions) a first-class case, not a hack.

### 2.3 `Metric` protocol

```python
class Metric(Protocol):
    spec: ClassVar[MetricSpec]
    def compute(
        self,
        real: pd.DataFrame | None,
        synthetic: pd.DataFrame,
        meta: Metadata,
    ) -> MetricResult: ...
```

### 2.4 `MetricResult`

```python
@dataclass
class MetricResult:
    spec: MetricSpec
    scalars: dict[str, float] | None = None          # None ⇒ failed
    per_column: dict[str, dict[str, float]] | None = None
    plot_payload: PlotPayload | None = None
    notes: list[str] = field(default_factory=list)   # warnings: skipped cols, NaN drops, etc.
    error: MetricError | None = None                 # populated on failure
```

### 2.5 `PlotPayload`

```python
class PlotPayload(Protocol):
    def render_matplotlib(self) -> bytes: ...        # PNG bytes
    def render_plotly(self) -> str: ...              # HTML fragment
```

Concrete payloads live alongside metrics: `HistogramOverlay`, `CorrelationDiffHeatmap`, `PCAScatter`, `DCRHistogram`, `TSTRBars`. Pre-baked palette — most metrics reuse two or three.

### 2.6 `Report`

```python
@dataclass
class Report:
    metadata: Metadata
    results: list[MetricResult]
    run_info: RunInfo                                # versions, seeds, timing
    @property
    def by_c(self) -> dict[str, list[MetricResult]]: ...
    @property
    def aggregated(self) -> dict[str, float]: ...    # per-C scalar summary

    def to_dict(self) -> dict: ...
    @classmethod
    def from_dict(cls, d: dict) -> "Report": ...     # round-trip with to_dict / to_json
    def to_json(self, path: str | Path) -> None: ...
    def to_html(self, path: str | Path, *, interactive: bool = False) -> None: ...
    def to_pdf(self, path: str | Path) -> None: ...
```

`aggregated` follows the paper's §"Aggregating Scores for Decision Thresholds": normalized mean of each C's constituent metrics. v0.1 uses **uniform weights only**; per-metric weighting is deferred to v0.1.x.

`to_dict` / `from_dict` round-tripping is required so JSON-on-disk scorecards can be re-loaded into Python (e.g., for diffs, dashboards, or the future API/UI consuming archived reports).

### 2.7 `MetricRegistry`

```python
class MetricRegistry:
    def register(self, cls: type[Metric]) -> type[Metric]: ...
    def select(
        self,
        *,
        tiers: Iterable[str],
        data_type: str,
        cs: Iterable[str] | None = None,
    ) -> list[type[Metric]]: ...

registry = MetricRegistry()                          # module-level singleton
```

User-defined custom metric:

```python
from syneva import registry, Metric, MetricSpec, MetricResult

@registry.register
class MyMetric:
    spec = MetricSpec(
        name="my_metric", c="congruence", tier="custom",
        data_types=frozenset({"static"}),
        requires_real=True, scope="per-column",
    )
    def compute(self, real, synthetic, meta) -> MetricResult: ...
```

### 2.8 `Runner` (internal orchestrator)

Loops over selected metrics, enforces `requires_real`, catches per-metric exceptions, assembles the `Report`. Not user-facing.

### 2.9 Public entry point

```python
def evaluate(
    real: pd.DataFrame | None,
    synthetic: pd.DataFrame,
    metadata: Metadata | None = None,
    *,
    tiers: Sequence[str] = ("core",),
    data_type: str = "static",
    cs: Sequence[str] | None = None,
    utility_tasks: Sequence[UtilityTask] | None = None,
    run_utility: bool = False,
    random_state: int = 42,
    nan_policy: Literal["drop", "explicit_na", "raise"] = "drop",
    interactive: bool = False,
) -> Report: ...
```

---

## 3. v0.1 metric inventory (static tabular)

16 core + 11 extended = 27 metrics. Every metric ships with a corresponding `tests/unit/test_<name>.py` and a `docs/metrics/<name>.md` describing what it measures, when to use it, and its primary citation.

### 3.1 Congruence (distribution alignment)

| Tier | Metric | Scope |
|---|---|---|
| core | KS statistic + p-value | per-column (numeric) |
| core | Total Variation Distance | per-column (categorical) |
| core | Wasserstein-1 / EMD | per-column (numeric) |
| core | Correlation difference (Pearson / Cramér's V / η²) | pairwise |
| core | pMSE (propensity MSE via logistic regression) | table-level |
| extended | Sliced Wasserstein | table-level |
| extended | Jensen-Shannon divergence | per-column |
| extended | C2ST (Classifier Two-Sample Test) | table-level |

### 3.2 Coverage (variability, range, novelty)

| Tier | Metric | Scope |
|---|---|---|
| core | Category coverage | per-column (categorical) |
| core | Range coverage | per-column (numeric) |
| core | Novelty rate (non-duplicate fraction) | table-level |
| core | Entropy ratio H(synth)/H(real) | per-column |
| extended | α-precision / β-recall (Alaa et al. 2022) | table-level |
| extended | Authenticity (Alaa et al. 2022) | table-level |
| extended | 2-D PCA scatter (visualization-only) | table-level |

### 3.3 Compliance (privacy / disclosure risk)

| Tier | Metric | Scope |
|---|---|---|
| core | DCR (distance to closest record) | table-level |
| core | NNDR (1st-NN / 2nd-NN distance ratio) | table-level |
| core | k-anonymity on sensitive columns (QI set = all columns with `sensitive=True`) | table-level |
| core | Hidden / identical match rate | table-level |
| extended | Membership inference attack AUC (shadow-model baseline) | table-level |
| extended | ε-DP ledger pass-through | metadata-driven |

### 3.4 Task-based ML utility

| Tier | Metric | Scope |
|---|---|---|
| core | TSTR (Train Synth, Test Real) | per `UtilityTask` |
| core | TRTR baseline (Train Real, Test Real) | per `UtilityTask` |
| core | Utility ratio TSTR / TRTR | per `UtilityTask` |
| extended | Multi-target sweep | per `UtilityTask` |
| extended | Feature-importance Spearman correlation | per `UtilityTask` |
| extended | Discriminative score (synth-vs-real classifier AUC) | table-level |

```python
@dataclass
class UtilityTask:
    target: str
    task_type: Literal["classification", "regression"] | None = None  # inferred from dtype
    features: list[str] | None = None                                  # default: all other non-ID columns
```

**Utility default behaviour:** when `utility_tasks=None`, `evaluate()` auto-suggests one task per non-ID column and stores the suggestion in `report.run_info`, but **runs nothing** unless `run_utility=True`. Avoids surprise multi-minute model training during routine fidelity checks.

---

## 4. Data flow

```
evaluate(real, synthetic, metadata=None, ...)

  1. Ingest
     ├── metadata = Metadata.infer(real if real is not None else synthetic) if None else metadata
     ├── validate(real, synthetic, metadata)            → raise SchemaError / MetadataError on mismatch
     └── if real is None and no synthetic-only metric available → raise SynevaError

  2. Select metrics
     └── registry.select(tiers=tiers, data_type=data_type, cs=cs)
         → filter by requires_real depending on whether real is provided
         → utility metrics only included if run_utility=True

  3. Execute (per metric, isolated)
     for cls in selected_metrics:
         try:    result = cls().compute(real, synthetic, metadata)
         except: result = MetricResult(spec=cls.spec, error=MetricError(...))

  4. Aggregate
     └── group results by C, compute per-C aggregate score (uniform weights in v0.1)

  5. Assemble Report
     ├── results, by_c (derived), aggregated (derived)
     └── run_info: syneva version, library versions, random_state, timing, warnings

  6. (optional, only on the user's explicit call) Render
     report.to_json(out / "scorecard.json")
     report.to_html(out / "scorecard.html", interactive=False)
     report.to_pdf(out / "scorecard.pdf")
```

### 4.1 Invariants

- **Metric isolation:** a metric raising an exception never aborts the run. Its `MetricResult` carries `error`; the HTML report renders it with a red badge and the exception's first line visible.
- **`run_info` is mandatory.** Captured automatically — never user-supplied — so two scorecards are diffable.
- **Determinism:** one `random_state` propagates to every metric using randomness (pMSE, C2ST, MIA, TSTR, PCA). Default 42, user-overridable.
- **No side effects in `evaluate()`.** Only `report.to_X()` writes to disk. The CLI is the one place a path triggers renders.

---

## 5. Error handling

**Principle:** fail fast on user errors, isolate per-metric failures, never silently drop data.

### 5.1 Error taxonomy

```python
class SynevaError(Exception): ...
class SchemaError(SynevaError): ...           # DataFrame shape/columns/dtypes
class MetadataError(SynevaError): ...         # Metadata conflicts with DataFrame
class RegistryError(SynevaError): ...         # unknown tier / metric / data_type
class MetricError(SynevaError): ...           # wraps per-metric failures (stored, not raised)
```

### 5.2 Fail-fast validation gates

| Failure | Where caught | Result |
|---|---|---|
| `synthetic` has columns `real` lacks (or vice versa) | Ingest | `SchemaError` with `extra=[…], missing=[…]` |
| Same column, incompatible dtype across real/synth | Ingest | `SchemaError` |
| Metadata declares `dtype=numeric` but column is `object` | Ingest | `MetadataError` |
| `metadata.primary_key` not in DataFrame | Ingest | `MetadataError` |
| `tiers=["foo"]` | `registry.select` | `RegistryError` |
| `data_type="longitudinal"` in v0.1 | `registry.select` | `RegistryError("longitudinal not yet implemented; v0.2 only")` |
| `real is None` and no synthetic-only metric in the selection | Ingest | `SynevaError("no runnable metrics for synthetic-only mode")` |

### 5.3 Isolated per-metric failure

Runner wraps every `cls().compute(...)` in `try/except`. On failure, populates `MetricResult.error`. Renderers show failed metrics with their exception message; they do not hide them.

### 5.4 NaN policy

Controlled by `evaluate(..., nan_policy=...)`. Default `"drop"`.

- **Numeric** + `"drop"`: per-metric drop of NaN rows for the active column; `notes` records the percentage dropped.
- **Categorical**: NaN becomes the explicit category `"__NA__"`; appears in TVD / category coverage like any other.
- **All-NaN column**: skipped for the active metric; recorded in `notes`.
- `"explicit_na"`: NaN treated as a value across all dtypes.
- `"raise"`: any NaN raises `SchemaError`.

### 5.5 Optional renderer dependencies

```toml
[project.optional-dependencies]
pdf = ["weasyprint>=62"]
plotly = ["plotly>=5.20"]
all = ["syneva[pdf,plotly]"]
```

`report.to_pdf()` and `interactive=True` check at runtime and raise `SynevaError("PDF rendering requires pip install 'syneva[pdf]'")` if the optional dep is missing. Base install stays light.

### 5.6 Warnings vs errors

- **Errors** (raise): things `evaluate` cannot meaningfully continue past.
- **Warnings** (`warnings.warn` + stored in `report.run_info.warnings`): "synthetic has 100 rows vs real's 1M; small-sample metrics may be noisy", "no `utility_tasks` declared and `run_utility=True`; auto-suggested target was `<col>`".

---

## 6. Testing strategy

Correctness of metric values is the highest-value test target. Wrong numbers are worse than crashes.

### 6.1 Test layers

| Layer | Asserts | Tools |
|---|---|---|
| Per-metric unit | Known-input → textbook-verified output | pytest |
| Property-based | Invariants every metric must satisfy | hypothesis |
| Integration / golden-report | Full `evaluate()` JSON snapshot stable across runs | pytest + json diff |
| Renderer smoke | `to_html` / `to_pdf` don't crash; key DOM elements present | beautifulsoup4 |
| CLI | `syn-eva evaluate` on fixtures writes expected files, exit 0 | pytest + subprocess |
| Type | Static contracts hold | pyright |
| Lint | Style + obvious bugs | ruff |

### 6.2 Auto-generated metric invariants

For every metric registered, the test framework generates these checks:

| Invariant | Applies to |
|---|---|
| `metric(real, real, meta).scalars ≈ ideal_score` | `requires_real=True` — identical inputs must score "indistinguishable" |
| `metric(real, shifted_synth, meta)` ranks worse than `metric(real, real, meta)` | Congruence / Coverage |
| `metric(real, real_subset, meta)` flags low privacy | Compliance |
| Column-order permutation invariance | per-column metrics |
| `Report.to_dict()` ↔ `Report.from_dict()` round-trip | every metric |
| Plot payload renders in both matplotlib and plotly without throwing | every metric with a payload |

### 6.3 Fixtures (small, versioned, in-repo)

```
tests/fixtures/
├── adult_income_real_500.parquet         # subset of UCI Adult
├── adult_income_syn_good_500.parquet     # high-quality synth, frozen
├── adult_income_syn_shifted_500.parquet  # deliberately shifted distributions
├── adult_income_syn_leaky_100.parquet    # contains 80 real rows verbatim
└── metadata.json
```

Plus hypothesis-driven generation for edge cases: empty DataFrames, single-row, all-NaN column, single-category categorical, extreme-skew numeric.

### 6.4 Golden reports

`tests/golden/*.scorecard.json` snapshots from fixed (real, syn, random_state) pairs. CI diffs produced JSON against golden. Regeneration is explicit via `pytest --update-golden` and reviewed in PRs — drift is visible, not silent.

### 6.5 Out of scope for v0.1

- Mutation testing
- Visual regression for plots (cross-platform pixel diffs are too noisy)
- Performance benchmarking gates (track informally with `pytest-benchmark` markers, no failure threshold)

### 6.6 CI gates (single GitHub Actions workflow on push and PR)

```yaml
- ruff check
- pyright src/syneva
- pytest -x --cov=syneva --cov-fail-under=85
```

85% coverage gate, excluding `syneva/render/` (HTML/PDF templating). Bump after v0.1 stabilizes.

---

## 7. Repo scaffolding

### 7.1 Tree (v0.1)

```
Syn_Eva/
├── pyproject.toml
├── uv.lock
├── README.md
├── LICENSE                                  # Apache-2.0
├── CHANGELOG.md
├── .gitignore
├── .python-version                          # 3.10
├── .pre-commit-config.yaml
├── .github/workflows/ci.yml
├── src/syneva/
│   ├── __init__.py                          # public surface (see §7.3)
│   ├── core/
│   │   ├── metadata.py
│   │   ├── metric.py
│   │   ├── plot.py
│   │   ├── registry.py
│   │   ├── runner.py
│   │   ├── report.py
│   │   ├── errors.py
│   │   └── run_info.py
│   ├── congruence/ {ks,tvd,wasserstein,correlation_diff,pmse}.py
│   │   └── extended/ {sliced_wasserstein,jsd,c2st}.py
│   ├── coverage/ {category_coverage,range_coverage,novelty,entropy_ratio}.py
│   │   └── extended/ {alpha_precision,authenticity,pca_scatter}.py
│   ├── compliance/ {dcr,nndr,k_anonymity,identical_match}.py
│   │   └── extended/ {mia,dp_ledger}.py
│   ├── utility/ {tstr,trtr,utility_ratio,task}.py
│   │   └── extended/ {multi_target,feature_importance,discriminative}.py
│   ├── constraint/    __init__.py only      # v0.2 placeholder
│   ├── completeness/  __init__.py only      # v0.2 placeholder
│   ├── comprehension/ __init__.py only      # v0.3 placeholder
│   ├── consistency/   __init__.py only      # v0.3 placeholder
│   ├── backends/
│   │   ├── static.py
│   │   ├── longitudinal.py                  # raises NotImplementedError("v0.2")
│   │   └── relational.py                    # raises NotImplementedError("v0.3")
│   ├── render/
│   │   ├── html/{templates/, renderer.py}
│   │   ├── pdf.py
│   │   └── json.py
│   └── cli/{__init__.py, main.py}
├── tests/
│   ├── conftest.py
│   ├── fixtures/ {*.parquet, metadata.json}
│   ├── golden/   {*.scorecard.json}
│   ├── unit/                                # per-metric correctness
│   ├── property/                            # hypothesis invariants
│   ├── integration/                         # full evaluate() + renderers
│   └── cli/
├── docs/
│   ├── superpowers/specs/                   # this file lives here
│   ├── metrics/                             # one .md per metric
│   └── quickstart.md
└── examples/
    ├── quickstart.ipynb
    └── adult_income_demo.py
```

### 7.2 `pyproject.toml` essentials

```toml
[project]
name = "syneva"
version = "0.1.0"
requires-python = ">=3.10"
license = "Apache-2.0"
authors = [{name = "Chang Sun", email = "chang.sun@maastrichtuniversity.nl"}]
description = "7 Cs scorecard for tabular synthetic data evaluation."
dependencies = [
  "pandas>=2.0",
  "numpy>=1.24",
  "scipy>=1.11",
  "scikit-learn>=1.3",
  "matplotlib>=3.7",
  "jinja2>=3.1",
  "typer>=0.12",
  "pydantic>=2.7",
]

[project.optional-dependencies]
pdf = ["weasyprint>=62"]
plotly = ["plotly>=5.20"]
all = ["syneva[pdf,plotly]"]

[project.scripts]
syn-eva = "syneva.cli.main:app"

[dependency-groups]
dev = ["pytest>=8.0", "pytest-cov>=5.0", "hypothesis>=6.0",
       "ruff>=0.5", "pyright>=1.1.380", "pre-commit>=3.7",
       "beautifulsoup4>=4.12"]

[tool.ruff]
line-length = 100
target-version = "py310"
[tool.ruff.lint]
select = ["E", "F", "I", "W", "UP", "B", "SIM", "RUF"]

[tool.pyright]
strict = ["src/syneva"]
pythonVersion = "3.10"

[tool.pytest.ini_options]
testpaths = ["tests"]
markers = ["slow"]
```

### 7.3 Public API surface (`syneva/__init__.py`)

```python
from syneva.core.metadata import Metadata, ColumnMetadata, ColumnType
from syneva.core.metric import Metric, MetricSpec, MetricResult
from syneva.core.registry import registry
from syneva.core.report import Report
from syneva.core.run_info import RunInfo
from syneva.core.runner import evaluate
from syneva.utility.task import UtilityTask
from syneva.core.errors import (
    SynevaError, SchemaError, MetadataError, RegistryError, MetricError,
)
from syneva._version import __version__
```

Everything else is implementation detail behind `syneva.core` / `syneva.<C>` / `syneva.render`.

### 7.4 Dev workflow

```bash
# one-time
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync --all-extras
uv run pre-commit install

# day-to-day
uv run pytest
uv run ruff check .
uv run pyright
uv run syn-eva evaluate --real real.csv --synthetic syn.csv --out report/
```

### 7.5 Versioning + release

- Semantic versioning. v0.x pre-stable; breaking changes allowed across minor versions.
- v0.1 ships static + 4 Cs + Task-based utility. v0.2 adds longitudinal + Constraint + Completeness. v0.3 adds relational + Comprehension + Consistency. v1.0 = stable API.
- Release: `uv build` → `uv publish` (TestPyPI first, then PyPI).
- CHANGELOG follows "keep-a-changelog" format.

---

## 8. References

- Zamzmi G, Subbaswamy A, Sizikova E, Margerrison E, Delfino JG, Badano A. *Scorecard for synthetic medical data evaluation.* Communications Engineering 4:130 (2025). DOI: 10.1038/s44172-025-00450-1.
- Alaa A, van Breugel B, Saveliev ES, van der Schaar M. *How Faithful is your Synthetic Data? Sample-level Metrics for Evaluating and Auditing Generative Models.* ICML 2022. (α-precision, β-recall, authenticity)
- Lopez-Paz D, Oquab M. *Revisiting Classifier Two-Sample Tests.* ICLR 2017. (C2ST)
- Snoke J, Raab GM, Nowok B, Dibben C, Slavkovic A. *General and specific utility measures for synthetic data.* JRSS-A (2018). (pMSE)
