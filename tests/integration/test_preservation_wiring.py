import numpy as np
import pandas as pd

import syneva
from syneva import SubgroupSpec
from syneva.core.metadata import ColumnMetadata, ColumnType, Metadata

_AUTO = {"rare_category_retention", "tail_coverage", "minority_class_density"}
_SUBGROUP = {"subgroup_fidelity", "minority_utility_gap", "minority_privacy_risk"}


def _df():
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "g": ["A"] * 160 + ["B"] * 40,
            "x": rng.normal(0, 1, 200),
        }
    )


def _meta():
    return Metadata(
        columns={
            "g": ColumnMetadata(name="g", dtype=ColumnType.CATEGORICAL),
            "x": ColumnMetadata(name="x", dtype=ColumnType.NUMERIC),
        }
    )


def test_core_tier_runs_no_preservation():
    rep = syneva.evaluate(_df(), _df(), _meta(), tiers=("core",))
    assert not {r.spec.name for r in rep.results} & (_AUTO | _SUBGROUP)


def test_auto_metrics_run_by_default_subgroup_metrics_dropped():
    rep = syneva.evaluate(_df(), _df(), _meta(), tiers=("core", "extended"))
    names = {r.spec.name for r in rep.results}
    assert names >= _AUTO
    assert not names & _SUBGROUP


def test_subgroup_metrics_run_with_specs():
    rep = syneva.evaluate(
        _df(),
        _df(),
        _meta(),
        tiers=("core", "extended"),
        subgroup_specs=[SubgroupSpec(name="b", conditions={"g": ["B"]})],
    )
    by = {r.spec.name: r for r in rep.results}
    assert set(by) >= _SUBGROUP
    for name in _AUTO | _SUBGROUP:
        assert by[name].error is None, f"{name}: {by[name].error}"
        assert 0.0 <= by[name].scalars["score"] <= 1.0


def test_benchmark_forwards_subgroup_specs():
    res = syneva.benchmark(
        _df(),
        {"cand": _df()},
        _meta(),
        tiers=("core", "extended"),
        subgroup_specs=[SubgroupSpec(name="b", conditions={"g": ["B"]})],
    )
    names = {r.spec.name for r in res.reports["cand"].results}
    assert names >= _SUBGROUP
    assert "preservation" in res.reports["cand"].aggregated
