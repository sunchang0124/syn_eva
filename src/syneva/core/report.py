from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
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
            usable = [r for r in rs if r.scalars is not None and r.error is None]
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
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, default=str))

    def to_html(self, path: str | Path, *, interactive: bool = False) -> None:
        from syneva.render.html.renderer import render_html

        Path(path).write_text(render_html(self, interactive=interactive))

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
        },
        "scalars": r.scalars,
        "per_column": r.per_column,
        "notes": r.notes,
        # MetricError.original (the causing exception) is not JSON-serializable
        # and is intentionally dropped; only the message string is persisted.
        "error": str(r.error) if r.error else None,
        "skip_reason": r.skip_reason,
    }


def _result_from_dict(d: dict) -> MetricResult:
    s = d["spec"]
    spec = MetricSpec(
        name=s["name"],
        c=s["c"],
        tier=s["tier"],
        data_types=frozenset(s["data_types"]),
        requires_real=s["requires_real"],
        scope=s["scope"],
    )
    err = MetricError(d["error"]) if d.get("error") else None
    return MetricResult(
        spec=spec,
        scalars=d.get("scalars"),
        per_column=d.get("per_column"),
        notes=d.get("notes") or [],
        error=err,
        skip_reason=d.get("skip_reason"),
    )


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
    if "score" in r.scalars:
        v = r.scalars["score"]
    else:
        first = next(iter(r.scalars.values()))
        v = 1.0 - min(1.0, max(0.0, first))
    return float(min(1.0, max(0.0, v)))
