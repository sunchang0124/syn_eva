import numpy as np
import pandas as pd
import pytest

from syneva import SubgroupSpec
from syneva.core.metadata import ColumnMetadata, ColumnType, Metadata
from syneva.preservation.minority_privacy_risk import MinorityPrivacyRisk


def _meta():
    return Metadata(
        columns={
            "g": ColumnMetadata(name="g", dtype=ColumnType.CATEGORICAL),
            "x": ColumnMetadata(name="x", dtype=ColumnType.NUMERIC),
        }
    )


def _real():
    rng = np.random.default_rng(1)
    return pd.DataFrame({"g": ["A"] * 80 + ["B"] * 20, "x": rng.normal(0, 1, 100)})


_SPEC = SubgroupSpec(name="b-group", conditions={"g": ["B"]})


def test_copied_subgroup_scores_low():
    real = _real()
    # synthetic copies the subgroup rows verbatim; everything else is far away
    far = real[real.g == "A"].copy()
    far["x"] = far["x"] + 50
    syn = pd.concat([far, real[real.g == "B"].copy()], ignore_index=True)
    res = MinorityPrivacyRisk(subgroup_specs=[_SPEC]).compute(real, syn, _meta())
    assert res.scalars["score"] < 0.1
    assert res.per_column["b-group"]["risk_ratio"] < 0.1


def test_uniformly_close_synthetic_scores_high():
    real = _real()
    syn = real.copy()
    syn["x"] = syn["x"] + 0.3  # same offset everywhere -> no concentration
    res = MinorityPrivacyRisk(subgroup_specs=[_SPEC]).compute(real, syn, _meta())
    assert res.scalars["score"] > 0.5


def test_worst_subgroup_drives_score():
    real = _real()
    far = real[real.g == "A"].copy()
    far["x"] = far["x"] + 50
    syn = pd.concat([far, real[real.g == "B"].copy()], ignore_index=True)
    safe = SubgroupSpec(name="a-group", conditions={"g": ["A"]})
    res = MinorityPrivacyRisk(subgroup_specs=[safe, _SPEC]).compute(real, syn, _meta())
    assert res.scalars["score"] < 0.1  # min over specs, not mean
    assert res.scalars["mean_risk_ratio"] > res.scalars["worst_risk_ratio"]


def test_tiny_subgroup_skipped():
    spec = SubgroupSpec(name="tiny", conditions={"x": (100.0, None)})
    real = _real()
    res = MinorityPrivacyRisk(subgroup_specs=[spec]).compute(real, real.copy(), _meta())
    assert res.scalars is None
    assert res.skip_reason
    assert any("tiny" in n for n in res.notes)


def test_tiny_rest_of_data_skipped():
    # 94 of 100 rows match, leaving 6 rows outside the subgroup: too few to
    # give a baseline.
    spec = SubgroupSpec(name="nearly-all", conditions={"x": (-1.5, None)})
    real = _real()
    assert int((real.x < -1.5).sum()) == 6
    res = MinorityPrivacyRisk(subgroup_specs=[spec]).compute(real, real.copy(), _meta())
    assert res.scalars is None
    assert res.skip_reason
    assert any("nearly-all" in n and "outside" in n for n in res.notes)


def test_no_specs_is_skipped():
    real = _real()
    res = MinorityPrivacyRisk().compute(real, real.copy(), _meta())
    assert res.scalars is None
    assert res.skip_reason


def test_capped_subgroup_majority_still_scores_low():
    # subgroup spans the majority of real rows and exceeds cap: without capping
    # members down, all kept rows would be members, the baseline would collapse
    # onto the subgroup, and a verbatim-copied subgroup would falsely score ~1.
    rng = np.random.default_rng(2)
    real = pd.DataFrame(
        {
            "g": ["B"] * 120 + ["A"] * 80,
            "x": rng.normal(0, 1, 200),
        }
    )
    far = real[real.g == "A"].copy()
    far["x"] = far["x"] + 50
    syn = pd.concat([far, real[real.g == "B"].copy()], ignore_index=True)
    res = MinorityPrivacyRisk(subgroup_specs=[_SPEC], cap=50).compute(real, syn, _meta())
    assert res.scalars["score"] < 0.1


def test_capped_specs_covering_all_rows_each_get_a_ratio():
    # Together the two specs cover every row, so no row is outside both, but
    # each spec still has the other as its baseline.
    rng = np.random.default_rng(3)
    real = pd.DataFrame({"g": ["A"] * 100 + ["B"] * 100, "x": rng.normal(0, 1, 200)})
    far = real[real.g == "A"].copy()
    far["x"] = far["x"] + 50
    syn = pd.concat([far, real[real.g == "B"].copy()], ignore_index=True)
    safe = SubgroupSpec(name="a-group", conditions={"g": ["A"]})
    res = MinorityPrivacyRisk(subgroup_specs=[safe, _SPEC], cap=100).compute(real, syn, _meta())
    assert set(res.per_column) == {"a-group", "b-group"}
    assert res.per_column["b-group"]["risk_ratio"] < 0.1


def test_capped_subgroup_spanning_all_rows_not_measurable():
    real = _real()
    every = SubgroupSpec(name="every", conditions={"g": ["A", "B"]})
    res = MinorityPrivacyRisk(subgroup_specs=[every], cap=40).compute(real, real.copy(), _meta())
    assert res.scalars is None
    assert res.skip_reason and "measurable" in res.skip_reason
    assert any("every" in n and "spans all kept rows" in n for n in res.notes)


@pytest.mark.parametrize("distance", ["euclidean", "gower"])
def test_capped_baseline_matches_uncapped_baseline(distance):
    # A 2% subgroup the synthetic data copies with noise. Capping keeps every
    # member but only ~2% of the others, so an unweighted baseline is half
    # subgroup and the concentrated risk would read as smaller than it is.
    rng = np.random.default_rng(0)
    cols = [f"x{i}" for i in range(5)]
    meta = Metadata(
        columns={
            "g": ColumnMetadata(name="g", dtype=ColumnType.CATEGORICAL),
            **{c: ColumnMetadata(name=c, dtype=ColumnType.NUMERIC) for c in cols},
        }
    )
    real = pd.DataFrame(rng.normal(0, 1, (10_000, 5)), columns=cols)
    real.insert(0, "g", ["B"] * 200 + ["A"] * 9_800)
    copied = real[real.g == "B"].copy()
    copied[cols] += rng.normal(0, 0.3, (200, 5))
    fresh = pd.DataFrame(rng.normal(0, 1, (200, 5)), columns=cols)
    fresh.insert(0, "g", "A")
    syn = pd.concat([copied, fresh], ignore_index=True)  # 400 rows: never capped

    def ratio(cap):
        res = MinorityPrivacyRisk(subgroup_specs=[_SPEC], distance=distance, cap=cap).compute(
            real, syn, meta
        )
        return res.per_column["b-group"]["risk_ratio"]

    assert ratio(cap=400) == pytest.approx(ratio(cap=10_000), abs=0.05)


def test_gower_backend_runs():
    real = _real()
    res = MinorityPrivacyRisk(subgroup_specs=[_SPEC], distance="gower").compute(
        real, real.copy(), _meta()
    )
    assert 0.0 <= res.scalars["score"] <= 1.0
