import pytest

import syneva
from syneva.compliance.extended.epsilon_identifiability import EpsilonIdentifiability
from syneva.core.errors import SynevaError
from syneva.coverage.extended.authenticity import Authenticity


def test_gower_distance_runs_end_to_end(real_df, syn_good_df, metadata):
    rep = syneva.evaluate(
        real_df, syn_good_df, metadata, tiers=("core", "extended"), distance="gower"
    )
    by = {r.spec.name: r for r in rep.results}
    for m in [
        "dcr",
        "nndr",
        "nn_adversarial_accuracy",
        "authenticity",
        "alpha_precision_beta_recall",
        "epsilon_identifiability",
    ]:
        assert m in by
        assert by[m].error is None, f"{m} errored under gower: {by[m].error}"


def test_invalid_distance_raises(real_df, syn_good_df, metadata):
    with pytest.raises(SynevaError, match="distance"):
        syneva.evaluate(real_df, syn_good_df, metadata, distance="manhattan")


@pytest.mark.parametrize(
    ("metric", "key"),
    [
        (Authenticity, "authenticity"),
        (EpsilonIdentifiability, "identifiability_risk"),
    ],
)
def test_one_synthetic_outlier_does_not_shift_gower_metrics(
    metric, key, same_distribution_with_outlier
):
    real, syn, outlier, meta = same_distribution_with_outlier
    clean = metric(distance="gower").compute(real, syn, meta).scalars[key]
    dirty = metric(distance="gower").compute(real, outlier, meta).scalars[key]
    assert dirty == pytest.approx(clean, abs=0.02)
