from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import pandas as pd

from syneva.core.errors import SynevaError
from syneva.core.metadata import Metadata
from syneva.core.presets import _UNSET
from syneva.core.report import Report, _normalize_scalars, is_scored
from syneva.core.runner import evaluate


def benchmark(
    real: pd.DataFrame | None,
    candidates: dict[str, pd.DataFrame],
    metadata: Metadata | None = None,
    *,
    holdout: pd.DataFrame | None = None,
    preset: str | None = None,
    tiers=_UNSET,
    cs=_UNSET,
    run_utility=_UNSET,
    utility_tasks=None,
    run_fairness=_UNSET,
    fairness_specs=None,
    subgroup_specs=None,
    distance=_UNSET,
    random_state: int = 42,
) -> BenchmarkResult:
    if not candidates:
        raise SynevaError("benchmark needs at least one candidate")
    ref_cols = (
        set(real.columns) if real is not None else set(next(iter(candidates.values())).columns)
    )
    for name, df in candidates.items():
        if set(df.columns) != ref_cols:
            raise SynevaError(f"candidate '{name}' columns must match real")
        if len(df) == 0:
            raise SynevaError(f"candidate '{name}' has no rows")
    reports: dict[str, Report] = {}
    for name, df in candidates.items():
        reports[name] = evaluate(
            real,
            df,
            metadata,
            holdout=holdout,
            preset=preset,
            tiers=tiers,
            cs=cs,
            run_utility=run_utility,
            utility_tasks=utility_tasks,
            run_fairness=run_fairness,
            fairness_specs=fairness_specs,
            subgroup_specs=subgroup_specs,
            distance=distance,
            random_state=random_state,
        )
    return BenchmarkResult(reports=reports, random_state=random_state)


@dataclass
class BenchmarkResult:
    reports: dict[str, Report]
    random_state: int = 42
    notes: list[str] = field(default_factory=list)
    # Derived from `reports` once, in __post_init__; `reports` is treated as fixed
    # after construction.
    _scores: dict[str, dict[str, float]] = field(init=False, repr=False, compare=False)
    _compared: list[str] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        self.notes = list(self.notes)  # don't add to a list the caller passed in
        self._scores = {
            name: {r.spec.name: _normalize_scalars(r) for r in rep.results if is_scored(r)}
            for name, rep in self.reports.items()
        }
        rows = [set(row) for row in self._scores.values() if row]
        self._compared = sorted(set.intersection(*rows)) if rows else []
        for note in self._comparison_notes():
            if note not in self.notes:
                self.notes.append(note)

    def _comparison_notes(self) -> list[str]:
        """One line per metric some candidate scored that isn't compared, naming the
        candidates that failed or skipped it, then one per candidate with nothing
        scored."""
        sm = self._scores
        compared = set(self._compared)
        scoring = [name for name, row in sm.items() if row]
        notes = []
        for m in self.metrics():
            if m in compared:
                continue
            lacking = []
            for name in scoring:
                if m not in sm[name]:
                    lacking.append(f"{self._status(name, m)} for {name}")
            notes.append(f"{m} not compared: {', '.join(lacking)}")
        notes += [
            f"{name} has no scored metrics and ranks last" for name, row in sm.items() if not row
        ]
        return notes

    def _status(self, candidate: str, metric: str) -> str:
        """Why `candidate` has no score for `metric`: it failed, was skipped, or isn't
        in the candidate's report at all."""
        for r in self.reports[candidate].results:
            if r.spec.name == metric:
                return "failed" if r.error is not None else "skipped"
        return "not run"

    @property
    def score_matrix(self) -> dict[str, dict[str, float]]:
        return {name: dict(row) for name, row in self._scores.items()}

    def compared_metrics(self) -> list[str]:
        """Metrics every candidate scored, which are the only ones candidates are
        compared on. A metric that failed or was skipped for one candidate would
        otherwise leave that candidate scored on an easier set than the others.
        Candidates with nothing scored are left out, so they don't empty the set;
        they rank last."""
        return list(self._compared)

    @property
    def c_scores(self) -> dict[str, dict[str, float]]:
        cmap = self._metric_c_map()
        out: dict[str, dict[str, float]] = {}
        for name, row in self._scores.items():
            by_c: dict[str, list[float]] = defaultdict(list)
            for m in self._compared:
                if m in row:
                    by_c[cmap[m]].append(row[m])
            out[name] = {c: sum(vs) / len(vs) for c, vs in sorted(by_c.items())}
        return out

    @property
    def overall(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for name, cs in self.c_scores.items():
            out[name] = (sum(cs.values()) / len(cs)) if cs else 0.0
        return out

    def metrics(self) -> list[str]:
        names: set[str] = set()
        for row in self._scores.values():
            names.update(row)
        return sorted(names)

    def c_dims(self) -> list[str]:
        dims: set[str] = set()
        for cs in self.c_scores.values():
            dims.update(cs)
        return sorted(dims)

    def _metric_c_map(self) -> dict[str, str]:
        out: dict[str, str] = {}
        for rep in self.reports.values():
            for r in rep.results:
                out[r.spec.name] = r.spec.c
        return out

    def _normalized_per_metric(self, normalization: str) -> dict[str, dict[str, float]]:
        from syneva.benchmark.normalize import normalize_across

        sm = self._scores
        out: dict[str, dict[str, float]] = {}
        for m in self.metrics():
            present = [c for c in self.reports if m in sm[c]]
            vals = [sm[c][m] for c in present]
            normed = normalize_across(vals, normalization)
            out[m] = dict(zip(present, normed, strict=True))
        return out

    def _candidate_key(
        self, candidate: str, metric_names: list[str], nm: dict[str, dict[str, float]]
    ) -> float:
        cmap = self._metric_c_map()
        by_c: dict[str, list[float]] = defaultdict(list)
        for m in metric_names:
            v = nm.get(m, {}).get(candidate)
            if v is not None:
                by_c[cmap[m]].append(v)
        if not by_c:
            return float("-inf")
        c_means = [sum(vs) / len(vs) for vs in by_c.values()]
        return sum(c_means) / len(c_means)

    def ranking(
        self, normalization: str = "absolute", by: str = "overall"
    ) -> list[tuple[int, str, float]]:
        cands = list(self.reports)
        if (
            len(cands) < 2
            and normalization != "absolute"
            and "single-candidate cohort: relative normalization is degenerate" not in self.notes
        ):
            self.notes.append("single-candidate cohort: relative normalization is degenerate")
        nm = self._normalized_per_metric(normalization)
        cmap = self._metric_c_map()
        if by == "overall":
            metric_names = self.compared_metrics()
        elif by in self.c_dims():
            metric_names = [m for m in self.compared_metrics() if cmap.get(m) == by]
        elif by in self.metrics():
            metric_names = [by]
        else:
            raise SynevaError(
                f"unknown ranking target by='{by}'; use 'overall', a C dimension, or a metric name"
            )
        keys = {c: self._candidate_key(c, metric_names, nm) for c in cands}
        order = sorted(cands, key=lambda c: (-keys[c], c))
        return [(i + 1, c, keys[c]) for i, c in enumerate(order)]

    def to_dict(self) -> dict:
        return {
            "reports": {name: rep.to_dict() for name, rep in self.reports.items()},
            "random_state": self.random_state,
        }

    @classmethod
    def from_dict(cls, d: dict) -> BenchmarkResult:
        return cls(
            reports={name: Report.from_dict(r) for name, r in d["reports"].items()},
            random_state=d.get("random_state", 42),
        )

    def to_json(self, path) -> None:
        import json
        from pathlib import Path

        Path(path).write_text(json.dumps(self.to_dict(), indent=2, default=str), encoding="utf-8")

    def to_html(self, path, *, normalization: str = "absolute", interactive: bool = False) -> None:
        from pathlib import Path

        from syneva.render.html.benchmark_renderer import render_leaderboard

        Path(path).write_text(
            render_leaderboard(self, normalization=normalization, interactive=interactive),
            encoding="utf-8",
        )

    @classmethod
    def from_json(cls, path) -> BenchmarkResult:
        import json
        from pathlib import Path

        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
