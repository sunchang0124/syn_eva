"""`evaluate(random_state=...)` controls every random component of a run (#42)."""

import numpy as np
import pandas as pd
import pytest

from syneva.compliance.dcr import DCR
from syneva.compliance.extended.attribute_disclosure import AttributeDisclosure
from syneva.compliance.extended.epsilon_identifiability import EpsilonIdentifiability
from syneva.compliance.extended.mia import MembershipInferenceAttack
from syneva.compliance.hitting_rate import HittingRate
from syneva.compliance.nndr import NNDR
from syneva.congruence.extended.c2st import C2ST
from syneva.congruence.extended.mmd import MMD
from syneva.congruence.extended.sliced_wasserstein import SlicedWasserstein
from syneva.congruence.pmse import PMSE
from syneva.core.metadata import ColumnMetadata, ColumnType, Metadata
from syneva.core.registry import MetricRegistry
from syneva.core.runner import evaluate_with
from syneva.coverage.extended.alpha_precision import AlphaPrecisionBetaRecall
from syneva.coverage.extended.authenticity import Authenticity
from syneva.coverage.extended.nn_adversarial_accuracy import NNAdversarialAccuracy
from syneva.coverage.extended.pca_scatter import PCAScatterMetric
from syneva.render.figures_pca import PCAScatter
from syneva.utility.extended.discriminative import DiscriminativeScore

_META = Metadata(
    columns={
        "x": ColumnMetadata(name="x", dtype=ColumnType.NUMERIC),
        "y": ColumnMetadata(name="y", dtype=ColumnType.NUMERIC),
        "c": ColumnMetadata(name="c", dtype=ColumnType.CATEGORICAL),
    }
)


def _frames(n: int, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)

    def frame(shift: float) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "x": rng.normal(shift, 1, n),
                "y": rng.normal(0, 1 + shift, n),
                "c": rng.choice(["a", "b", "c"], n),
            }
        )

    return frame(0.0), frame(0.3)


def _run(cls, real, syn, seed, **kwargs):
    """Run one metric through evaluate() on its own registry."""
    reg = MetricRegistry()
    reg.register(cls)
    report = evaluate_with(
        reg,
        real=real,
        synthetic=syn,
        metadata=_META,
        tiers=("core", "extended"),
        # Utility metrics only run when asked for; harmless for the others.
        run_utility=cls.spec.c == "utility",
        random_state=seed,
        **kwargs,
    )
    (result,) = report.results
    assert result.scalars is not None, result.skip_reason
    return result.scalars


def _varies_with_seed(cls, real, syn, **kwargs) -> bool:
    # Several seeds, not two: a coarse scalar such as a hit rate can tie between two
    # different subsamples by chance.
    results = [_run(cls, real, syn, seed, **kwargs) for seed in range(4)]
    return any(r != results[0] for r in results[1:])


@pytest.mark.parametrize(
    "cls", [C2ST, pytest.param(DiscriminativeScore, marks=pytest.mark.slow), SlicedWasserstein]
)
def test_seed_changes_always_random_metrics(cls):
    real, syn = _frames(300)
    assert _varies_with_seed(cls, real, syn)


@pytest.mark.parametrize(("cls", "n"), [(MMD, 800), (HittingRate, 3000)])
def test_seed_changes_subsampling_above_the_cap(cls, n):
    # Each metric subsamples only above its row cap (500 for MMD, 2000 for the
    # hitting rate), so the tables are well over it.
    real, syn = _frames(n)
    assert _varies_with_seed(cls, real, syn)


@pytest.mark.parametrize(
    ("cls", "n"),
    [
        (C2ST, 300),
        (DiscriminativeScore, 300),
        (SlicedWasserstein, 300),
        (PMSE, 300),
        (MMD, 800),
        (HittingRate, 2400),
    ],
)
def test_standalone_run_matches_evaluate_with_the_same_seed(cls, n):
    real, syn = _frames(n)
    standalone = cls(random_state=5).compute(real, syn, _META).scalars
    assert standalone == _run(cls, real, syn, 5)


def test_pca_scatter_matches_evaluate_with_the_same_seed():
    # A plot-only metric: compare the projected points rather than scalars.
    real, syn = _frames(300)
    standalone = PCAScatterMetric(random_state=5).compute(real, syn, _META).plot_payload
    reg = MetricRegistry()
    reg.register(PCAScatterMetric)
    report = evaluate_with(
        reg, real=real, synthetic=syn, metadata=_META, tiers=("extended",), random_state=5
    )
    in_run = report.results[0].plot_payload
    assert isinstance(standalone, PCAScatter)
    assert isinstance(in_run, PCAScatter)
    np.testing.assert_array_equal(standalone.real_xy, in_run.real_xy)
    np.testing.assert_array_equal(standalone.synthetic_xy, in_run.synthetic_xy)


# Every metric that gets its distances from Neighbors and supports gower.
_GOWER_NEIGHBORS_METRICS = [
    DCR,
    NNDR,
    EpsilonIdentifiability,
    AlphaPrecisionBetaRecall,
    Authenticity,
    NNAdversarialAccuracy,
]


@pytest.mark.slow
def test_seed_changes_gower_subsampling():
    # Gower Neighbors subsample real and synthetic above 2000 rows.
    real, syn = _frames(3000)
    assert _varies_with_seed(DCR, real, syn, distance="gower")


@pytest.mark.slow
@pytest.mark.parametrize("cls", _GOWER_NEIGHBORS_METRICS)
def test_neighbors_metrics_match_standalone_runs(cls):
    # Above the cap, a standalone run subsamples like evaluate() only if the metric
    # passes its seed on to the Neighbors it builds.
    real, syn = _frames(2400)
    standalone = cls(distance="gower", random_state=5).compute(real, syn, _META).scalars
    assert standalone == _run(cls, real, syn, 5, distance="gower")


def test_mia_takes_the_run_seed():
    real, syn = _frames(200)
    holdout, _ = _frames(80, seed=1)
    standalone = (
        MembershipInferenceAttack(holdout=holdout, random_state=5).compute(real, syn, _META).scalars
    )
    assert standalone == _run(MembershipInferenceAttack, real, syn, 5, holdout=holdout)


def test_attribute_disclosure_takes_the_run_seed():
    meta = Metadata(
        columns={
            **_META.columns,
            "c": ColumnMetadata(name="c", dtype=ColumnType.CATEGORICAL, sensitive=True),
        }
    )
    real, syn = _frames(200)
    standalone = AttributeDisclosure(random_state=5).compute(real, syn, meta).scalars
    reg = MetricRegistry()
    reg.register(AttributeDisclosure)
    report = evaluate_with(
        reg, real=real, synthetic=syn, metadata=meta, tiers=("extended",), random_state=5
    )
    assert standalone == report.results[0].scalars
