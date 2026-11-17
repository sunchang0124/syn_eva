"""`evaluate(metric_options=...)` reaches the real metrics in the default registry (#54)."""

import numpy as np
import pandas as pd
import pytest

import syneva
from syneva.congruence.ci_overlap import CIOverlap
from syneva.congruence.pmse import PMSE
from syneva.core.errors import SynevaError


def _by_name(report, name):
    return next(r for r in report.results if r.spec.name == name)


def _sized_pair():
    # Synthetic set ten times the real one, so the two ci_overlap modes disagree:
    # the default narrows the synthetic interval, use_real_se does not.
    rng = np.random.default_rng(0)
    real = pd.DataFrame({"x": rng.normal(size=200), "y": rng.normal(size=200)})
    synthetic = pd.DataFrame({"x": rng.normal(size=2000), "y": rng.normal(size=2000)})
    return real, synthetic


def test_ci_overlap_use_real_se_through_evaluate():
    real, synthetic = _sized_pair()
    meta = syneva.Metadata.infer(real)
    rep = syneva.evaluate(
        real, synthetic, meta, metric_options={"ci_overlap": {"use_real_se": True}}
    )
    expected = CIOverlap(use_real_se=True).compute(real, synthetic, meta)
    default = CIOverlap().compute(real, synthetic, meta)
    got = _by_name(rep, "ci_overlap")
    assert got.scalars == expected.scalars
    assert got.scalars != default.scalars


def test_ci_overlap_default_through_evaluate():
    real, synthetic = _sized_pair()
    meta = syneva.Metadata.infer(real)
    rep = syneva.evaluate(real, synthetic, meta)
    assert _by_name(rep, "ci_overlap").scalars == CIOverlap().compute(real, synthetic, meta).scalars


def test_pmse_n_permutations_through_evaluate():
    real, synthetic = _sized_pair()
    meta = syneva.Metadata.infer(real)
    rep = syneva.evaluate(
        real, synthetic, meta, random_state=3, metric_options={"pmse": {"n_permutations": 2}}
    )
    expected = PMSE(random_state=3, n_permutations=2).compute(real, synthetic, meta)
    default = PMSE(random_state=3).compute(real, synthetic, meta)
    got = _by_name(rep, "pmse")
    assert got.scalars == expected.scalars
    assert got.scalars != default.scalars


def test_pmse_rejected_n_permutations_fails_only_pmse():
    real, synthetic = _sized_pair()
    rep = syneva.evaluate(real, synthetic, metric_options={"pmse": {"n_permutations": 0}})
    pmse = _by_name(rep, "pmse")
    assert pmse.error is not None
    assert "n_permutations must be at least 1" in str(pmse.error)
    assert _by_name(rep, "ci_overlap").error is None


def test_runner_controlled_option_on_real_metric_raises():
    real, synthetic = _sized_pair()
    with pytest.raises(
        SynevaError, match=r"'dcr' option 'distance'.*evaluate\(\)'s own 'distance'"
    ):
        syneva.evaluate(real, synthetic, metric_options={"dcr": {"distance": "gower"}})
