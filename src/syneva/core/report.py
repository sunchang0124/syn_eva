from __future__ import annotations

import json
import math
import numbers
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path

from syneva.core.errors import MetricError
from syneva.core.metadata import ColumnMetadata, ColumnType, Metadata
from syneva.core.metric import MetricResult, MetricSpec
from syneva.core.run_info import RunInfo


@dataclass
class Report:
    metadata: Metadata
    results: list[MetricResult]
    run_info: RunInfo

    @property
    def by_c(self) -> dict[str, list[MetricResult]]:
        out: dict[str, list[MetricResult]] = defaultdict(list)
        for r in self.results:
            out[r.spec.c].append(r)
        return dict(out)

    @property
    def aggregated(self) -> dict[str, float]:
        scores: dict[str, float] = {}
        for c, rs in self.by_c.items():
            usable = [r for r in rs if is_scored(r)]
            if not usable:
                continue
            per = [_normalize_scalars(r) for r in usable]
            scores[c] = sum(per) / len(per)
        return scores

    def to_dict(self) -> dict:
        return {
            "metadata": _metadata_to_dict(self.metadata),
            "results": [_result_to_dict(r) for r in self.results],
            "run_info": self.run_info.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> Report:
        return cls(
            metadata=_metadata_from_dict(d["metadata"]),
            results=[_result_from_dict(r) for r in d["results"]],
            run_info=RunInfo(**d["run_info"]),
        )

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(dumps_strict_json(self.to_dict()), encoding="utf-8")

    def to_html(self, path: str | Path, *, interactive: bool = False) -> None:
        from syneva.render.html.renderer import render_html

        Path(path).write_text(render_html(self, interactive=interactive), encoding="utf-8")

    def to_pdf(self, path: str | Path) -> None:
        from syneva.render.pdf import render_pdf

        render_pdf(self, path)


def _metadata_to_dict(m: Metadata) -> dict:
    return {
        "columns": {
            n: {
                "name": c.name,
                "dtype": c.dtype.value,
                "sensitive": c.sensitive,
                "quasi_identifier": c.quasi_identifier,
                "categories": c.categories,
                "value_range": list(c.value_range) if c.value_range else None,
            }
            for n, c in m.columns.items()
        },
        "primary_key": m.primary_key,
    }


def _metadata_from_dict(d: dict) -> Metadata:
    cols = {
        n: ColumnMetadata(
            name=c["name"],
            dtype=ColumnType(c["dtype"]),
            sensitive=c.get("sensitive", False),
            quasi_identifier=c.get("quasi_identifier", False),
            categories=c.get("categories"),
            value_range=tuple(c["value_range"]) if c.get("value_range") else None,
        )
        for n, c in d["columns"].items()
    }
    return Metadata(columns=cols, primary_key=d.get("primary_key"))


def _result_to_dict(r: MetricResult) -> dict:
    # plot_payload is intentionally not serialized: it is a non-JSON render
    # artefact (e.g. a matplotlib Figure) and has no place in the JSON report.
    return {
        "spec": {
            "name": r.spec.name,
            "c": r.spec.c,
            "tier": r.spec.tier,
            "data_types": sorted(r.spec.data_types),
            "requires_real": r.spec.requires_real,
            "scope": r.spec.scope,
            "scoring": r.spec.scoring,
        },
        "scalars": r.scalars,
        "per_column": r.per_column,
        "notes": r.notes,
        # MetricError.original (the causing exception) is not JSON-serializable
        # and is intentionally dropped; only the message string is persisted.
        "error": str(r.error) if r.error else None,
        "skip_reason": r.skip_reason,
    }


def _is_non_finite(x) -> bool:
    return (
        isinstance(x, numbers.Real) and not isinstance(x, numbers.Integral) and not math.isfinite(x)
    )


def _non_finite_key(k):
    # Keys are strings in JSON, so "NaN" / "Infinity" keys are valid; json.dumps has
    # always spelled them that way, but allow_nan=False would make them an error.
    if not _is_non_finite(k):
        return k
    if math.isnan(k):
        return "NaN"
    return "Infinity" if k > 0 else "-Infinity"


def _non_finite_to_none(obj):
    """Copy of ``obj`` with every NaN or infinite value replaced by None."""
    if isinstance(obj, dict):
        return {_non_finite_key(k): _non_finite_to_none(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_non_finite_to_none(v) for v in obj]
    return None if _is_non_finite(obj) else obj


def dumps_strict_json(d: dict) -> str:
    """Serialize a report or benchmark dict as valid JSON (RFC 8259).

    NaN and infinity have no JSON form; the default writer emits bare ``NaN`` /
    ``Infinity`` tokens that strict parsers reject, so they are written as null.
    ``allow_nan=False`` makes any value that slips through fail loudly instead.
    """
    return json.dumps(_non_finite_to_none(d), indent=2, default=str, allow_nan=False)


def _nulls_to_nan(values: dict) -> dict:
    """Copy of a flat ``{name: value}`` mapping with null values read as NaN."""
    return {k: math.nan if v is None else v for k, v in values.items()}


def _result_from_dict(d: dict) -> MetricResult:
    s = d["spec"]
    spec = MetricSpec(
        name=s["name"],
        c=s["c"],
        tier=s["tier"],
        data_types=frozenset(s["data_types"]),
        requires_real=s["requires_real"],
        scope=s["scope"],
        scoring=s.get("scoring", True),
    )
    err = MetricError(d["error"]) if d.get("error") else None
    scalars = d.get("scalars")
    per_column = d.get("per_column")
    # Files store NaN and infinity as null (#109); read them back as NaN.
    if scalars is not None:
        scalars = _nulls_to_nan(scalars)
    if per_column is not None:
        per_column = {
            col: _nulls_to_nan(vals) if isinstance(vals, dict) else vals
            for col, vals in per_column.items()
        }
    # A saved file can hold a non-finite score (written before #76, or edited), so
    # loaded results get the same check as freshly computed ones.
    return reject_non_finite_score(
        MetricResult(
            spec=spec,
            scalars=scalars,
            per_column=per_column,
            notes=d.get("notes") or [],
            error=err,
            skip_reason=d.get("skip_reason"),
        )
    )


def _has_score(r: MetricResult) -> bool:
    """A scoring metric that ran and reported scalars, whatever their values."""
    return r.spec.scoring and r.scalars is not None and r.error is None


def _non_finite_score(r: MetricResult) -> float | None:
    """The aggregated value if it is a NaN or infinite number, else None.

    Only numbers are checked: other malformed values are left for aggregation to
    reject, as they were before this check existed.
    """
    if not r.scalars:
        return None
    v = _aggregated_scalar(r)
    if isinstance(v, (int, float)) and not math.isfinite(v):
        return float(v)
    return None


def reject_non_finite_score(r: MetricResult) -> MetricResult:
    """Turn a result whose aggregated value is NaN or infinite into a failed metric.

    Aggregation clamps scores with min/max, which turn NaN into 0 or 1 depending
    on argument order, so a non-finite score would count silently. Reporting it
    as a failure shows it in the report and excludes it from aggregates.
    Non-finite secondary scalars (e.g. a NaN p-value) are kept.
    """
    if not _has_score(r) or (v := _non_finite_score(r)) is None:
        return r
    return replace(
        r, scalars=None, error=MetricError(f"{r.spec.name} returned a non-finite score ({v})")
    )


def is_scored(r: MetricResult) -> bool:
    """Whether a result counts toward dimension scores and benchmark rankings.

    Skipped (scalars=None), failed, and non-scoring results are excluded, and so is
    a non-finite aggregated value, which the clamp in aggregation would otherwise
    count as 0 or 1.
    """
    return _has_score(r) and _non_finite_score(r) is None


def _normalize_scalars(r: MetricResult) -> float:
    """Return a [0,1] score (1=ideal) for one MetricResult.

    v0.1 convention: metrics MUST satisfy exactly one of:
      - expose ``scalars["score"]`` in [0,1] where 1 = ideal (higher is better), OR
      - expose a single scalar that is a distance in [0,1] where 0 = ideal;
        this path inverts via ``1 - value``.
    Any other layout will aggregate incorrectly.
    """
    assert r.scalars is not None
    # Convention: metrics expose a "score" key in [0,1]. If not, fall back to
    # 1.0 - min(1.0, max(0.0, first_value)) for distance-like metrics.
    v = _aggregated_scalar(r)
    if "score" not in r.scalars:
        v = 1.0 - min(1.0, max(0.0, v))
    return float(min(1.0, max(0.0, v)))


def _aggregated_scalar(r: MetricResult) -> float:
    """Return the raw scalar aggregation reads: ``score`` if present, otherwise the
    first scalar (a distance)."""
    assert r.scalars is not None
    if "score" in r.scalars:
        return r.scalars["score"]
    return next(iter(r.scalars.values()))
