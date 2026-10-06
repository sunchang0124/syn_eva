import numpy as np
import pandas as pd

from syneva import SubgroupSpec
from syneva.core.metadata import ColumnMetadata, ColumnType, Metadata
from syneva.preservation.subgroup_fidelity import SubgroupFidelity


def _meta():
    return Metadata(
        columns={
            "g": ColumnMetadata(name="g", dtype=ColumnType.CATEGORICAL),
            "x": ColumnMetadata(name="x", dtype=ColumnType.NUMERIC),
        }
    )


def _real():
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "g": ["A"] * 160 + ["B"] * 40,
            "x": np.concatenate([rng.normal(0, 1, 160), rng.normal(5, 1, 40)]),
        }
    )


_SPEC = SubgroupSpec(name="b-group", conditions={"g": ["B"]})


def test_identical_scores_high():
    res = SubgroupFidelity(subgroup_specs=[_SPEC]).compute(_real(), _real().copy(), _meta())
    assert res.scalars["score"] > 0.95
    assert res.per_column["b-group"]["share_real"] == 0.2


def test_erased_subgroup_scores_zero():
    syn = pd.DataFrame({"g": ["A"] * 200, "x": np.zeros(200)})
    res = SubgroupFidelity(subgroup_specs=[_SPEC]).compute(_real(), syn, _meta())
    assert res.per_column["b-group"]["score"] == 0.0
    assert res.scalars["score"] == 0.0
    assert any("erased" in n for n in res.notes)


def test_shrunk_subgroup_drops_share_term():
    real = _real()
    syn = pd.concat([real[real.g == "A"], real[real.g == "B"].head(20)], ignore_index=True)
    res = SubgroupFidelity(subgroup_specs=[_SPEC]).compute(real, syn, _meta())
    # share_syn ~ 20/180 = 0.111 vs share_real 0.2 -> share term ~0.556; shape
    # stays high (same-distribution subsample) but KS noise on n=20 is real
    assert 0.6 < res.scalars["score"] < 0.9


def test_shifted_subgroup_drops_shape_term():
    real = _real()
    syn = real.copy()
    syn.loc[syn.g == "B", "x"] = syn.loc[syn.g == "B", "x"] + 10
    res = SubgroupFidelity(subgroup_specs=[_SPEC]).compute(real, syn, _meta())
    assert res.per_column["b-group"]["shape_score"] < 0.6
    assert res.per_column["b-group"]["share_real"] == 0.2


def test_tiny_real_subgroup_skipped():
    spec = SubgroupSpec(name="tiny", conditions={"x": (100.0, None)})
    res = SubgroupFidelity(subgroup_specs=[spec]).compute(_real(), _real().copy(), _meta())
    assert res.scalars is None
    assert res.skip_reason
    assert any("tiny" in n for n in res.notes)


def test_invalid_spec_skipped_with_note():
    spec = SubgroupSpec(name="bad", conditions={"g": (0, 1)})
    res = SubgroupFidelity(subgroup_specs=[spec]).compute(_real(), _real().copy(), _meta())
    assert res.scalars is None
    assert res.skip_reason
    assert any("bad" in n for n in res.notes)


def test_no_specs_is_skipped():
    res = SubgroupFidelity().compute(_real(), _real().copy(), _meta())
    assert res.scalars is None
    assert res.skip_reason
