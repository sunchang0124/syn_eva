"""Metrics that cannot run must be reported as skipped, never as a perfect score."""

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import syneva
from syneva.benchmark.engine import BenchmarkResult
from syneva.compliance.extended.dp_ledger import DPLedger
from syneva.congruence.pmse import PMSE
from syneva.core.metadata import Metadata
from syneva.core.metric import MetricResult, MetricSpec
from syneva.core.report import Report
from syneva.coverage.extended.pca_scatter import PCAScatterMetric
from syneva.render.html.renderer import render_html

SRC = Path(__file__).resolve().parents[3] / "src" / "syneva"


def _constant_score_returns() -> list[str]:
    hits = []
    for path in sorted(SRC.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Call) and getattr(node.func, "id", None) == "MetricResult"
            ):
                continue
            scalars = next((k.value for k in node.keywords if k.arg == "scalars"), None)
            if not isinstance(scalars, ast.Dict):
                continue
            for key, value in zip(scalars.keys, scalars.values, strict=True):
                is_score = isinstance(key, ast.Constant) and key.value == "score"
                if is_score and isinstance(value, ast.Constant):
                    hits.append(f"{path.relative_to(SRC)}:{node.lineno}")
    return hits


def test_no_metric_returns_a_hardcoded_score():
    # A hardcoded score is how skip paths used to fail open; use skip_reason instead.
    assert _constant_score_returns() == []


def _bool_datetime_frame() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "flag": rng.integers(0, 2, 200).astype(bool),
            "when": pd.date_range("2020-01-01", periods=200),
        }
    )


def test_unencodable_copy_is_not_rated_private():
    real = _bool_datetime_frame()
    rep = syneva.evaluate(real, real.copy(), Metadata.infer(real), preset="full")
    by = {r.spec.name: r for r in rep.results}
    for name in ("dcr", "nndr", "k_anonymity", "hitting_rate", "epsilon_identifiability"):
        assert by[name].scalars is None, name
        assert by[name].skip_reason, name
    # identical_match_rate is the only compliance metric that can run, and it sees a copy
    assert rep.aggregated["compliance"] == 0.0


def test_full_preset_without_fairness_specs_has_no_fairness_score(real_df, syn_good_df, metadata):
    rep = syneva.evaluate(real_df, syn_good_df, metadata, preset="full")
    parity = next(r for r in rep.results if r.spec.name == "statistical_parity")
    assert parity.scalars is None
    assert parity.skip_reason
    assert "fairness" not in rep.aggregated


def test_pca_scatter_is_non_scoring(real_df, syn_good_df, metadata):
    assert PCAScatterMetric.spec.scoring is False
    r = PCAScatterMetric().compute(real_df, syn_good_df, metadata)
    assert r.scalars is None
    assert r.plot_payload is not None


def test_dp_ledger_is_non_scoring_and_skips_without_a_ledger():
    assert DPLedger.spec.scoring is False
    syn = pd.DataFrame({"x": [1.0, 2.0]})
    meta = Metadata.infer(syn)
    assert DPLedger().compute(None, syn, meta).skip_reason
    syn.attrs["dp_ledger"] = {"epsilon": 3.0, "delta": 1e-5}
    r = DPLedger().compute(None, syn, meta)
    assert r.scalars == {"epsilon": 3.0, "delta": 1e-5}


def _spec(name: str, c: str = "compliance", scoring: bool = True) -> MetricSpec:
    return MetricSpec(
        name=name,
        c=c,
        tier="core",
        data_types=frozenset({"static"}),
        requires_real=True,
        scope="table-level",
        scoring=scoring,
    )


def _report(results: list[MetricResult]) -> Report:
    from syneva.core.run_info import RunInfo

    return Report(
        metadata=Metadata(columns={}), results=results, run_info=RunInfo.capture(random_state=0)
    )


def test_non_scoring_results_are_excluded_from_aggregates_and_ranking():
    rep = _report(
        [
            MetricResult(spec=_spec("real"), scalars={"score": 0.2}),
            MetricResult(spec=_spec("info", scoring=False), scalars={"epsilon": 0.0}),
        ]
    )
    assert rep.aggregated == {"compliance": pytest.approx(0.2)}
    assert BenchmarkResult(reports={"g": rep}).score_matrix == {"g": {"real": 0.2}}


def test_scoring_flag_round_trips_and_defaults_to_true():
    rep = _report([MetricResult(spec=_spec("info", scoring=False), scalars={"epsilon": 1.0})])
    d = rep.to_dict()
    assert Report.from_dict(d).results[0].spec.scoring is False
    del d["results"][0]["spec"]["scoring"]
    assert Report.from_dict(d).results[0].spec.scoring is True


def test_actual_mode_renders_results_without_a_score(real_df, syn_good_df, metadata):
    pca = PCAScatterMetric().compute(real_df, syn_good_df, metadata)
    skipped = MetricResult(spec=_spec("k_anonymity"), skip_reason="no sensitive columns")
    html = render_html(_report([pca, skipped]), score_mode="actual")
    assert "Skipped" in html


def test_pmse_classifier_failure_is_an_error_not_a_score(
    real_df, syn_good_df, metadata, monkeypatch
):
    def boom(*args, **kwargs):
        raise ValueError("solver diverged")

    monkeypatch.setattr("syneva.congruence.pmse.LogisticRegression.fit", boom)
    with pytest.raises(ValueError, match="solver diverged"):
        PMSE().compute(real_df, syn_good_df, metadata)
