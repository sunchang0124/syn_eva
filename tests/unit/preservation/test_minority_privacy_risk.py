import numpy as np
import pandas as pd

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
    assert res.scalars["score"] == 1.0
    assert any("tiny" in n for n in res.notes)


def test_no_specs_scores_one_with_note():
    real = _real()
    res = MinorityPrivacyRisk().compute(real, real.copy(), _meta())
    assert res.scalars["score"] == 1.0
    assert res.notes


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


def test_gower_backend_runs():
    real = _real()
    res = MinorityPrivacyRisk(subgroup_specs=[_SPEC], distance="gower").compute(
        real, real.copy(), _meta()
    )
    assert 0.0 <= res.scalars["score"] <= 1.0
