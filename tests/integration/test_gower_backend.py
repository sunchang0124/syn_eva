import pytest

import syneva
from syneva.core.errors import SynevaError


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
