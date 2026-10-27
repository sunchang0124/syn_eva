import pytest

import syneva
import syneva.core.neighbors as neighbors
from syneva.compliance.dcr import DCR
from syneva.compliance.extended.epsilon_identifiability import EpsilonIdentifiability
from syneva.compliance.nndr import NNDR
from syneva.coverage.extended.alpha_precision import AlphaPrecisionBetaRecall
from syneva.coverage.extended.authenticity import Authenticity
from syneva.coverage.extended.nn_adversarial_accuracy import NNAdversarialAccuracy

# Every metric that gets its Neighbors from the runner (#46).
_DISTANCE_METRICS = [
    DCR,
    NNDR,
    EpsilonIdentifiability,
    AlphaPrecisionBetaRecall,
    Authenticity,
    NNAdversarialAccuracy,
]


def _holdout(real_df, with_holdout):
    return real_df.sample(frac=0.3, random_state=2) if with_holdout else None


def _evaluate(real_df, syn_df, metadata, distance, holdout, **kwargs):
    return syneva.evaluate(
        real_df,
        syn_df,
        metadata,
        tiers=("core", "extended"),
        distance=distance,
        holdout=holdout,
        **kwargs,
    )


@pytest.mark.slow
@pytest.mark.parametrize(("with_holdout", "expected"), [(False, 3), (True, 5)])
def test_gower_builds_each_needed_matrix_once(
    monkeypatch, real_df, syn_good_df, metadata, with_holdout, expected
):
    calls = []
    original = neighbors.gower_matrix

    def spy(a, b, *args, **kwargs):
        calls.append((len(a), len(b)))
        return original(a, b, *args, **kwargs)

    monkeypatch.setattr(neighbors, "gower_matrix", spy)
    holdout = _holdout(real_df, with_holdout)
    _evaluate(real_df, syn_good_df, metadata, "gower", holdout)
    # Without a holdout: real x real, real x synthetic, synthetic x synthetic. With
    # one, the holdout instance adds real x synthetic and holdout x real, and never
    # holdout x holdout.
    assert len(calls) == expected
    if holdout is not None:
        assert (len(holdout), len(holdout)) not in calls


@pytest.mark.slow
@pytest.mark.parametrize(("with_holdout", "expected"), [(False, 1), (True, 2)])
def test_euclidean_encodes_once_per_shared_instance(
    monkeypatch, real_df, syn_good_df, metadata, with_holdout, expected
):
    calls = []

    def spying(original):
        def spy(*args, **kwargs):
            # attribute_disclosure builds its own Neighbors over quasi-identifiers only.
            if any(arg is metadata for arg in (*args, *kwargs.values())):
                calls.append(original.__name__)
            return original(*args, **kwargs)

        return spy

    monkeypatch.setattr(neighbors, "encode_pair", spying(neighbors.encode_pair))
    monkeypatch.setattr(neighbors, "encode_frames", spying(neighbors.encode_frames))
    _evaluate(real_df, syn_good_df, metadata, "euclidean", _holdout(real_df, with_holdout))
    assert len(calls) == expected


def test_no_neighbors_without_a_distance_metric(monkeypatch, real_df, syn_good_df, metadata):
    built = []
    original_init = neighbors.Neighbors.__init__

    def spy(self, *args, **kwargs):
        built.append(1)
        original_init(self, *args, **kwargs)

    monkeypatch.setattr(neighbors.Neighbors, "__init__", spy)
    syneva.evaluate(real_df, syn_good_df, metadata, cs=["congruence"])
    assert built == []


@pytest.mark.slow
@pytest.mark.parametrize("distance", ["euclidean", "gower"])
@pytest.mark.parametrize("with_holdout", [False, True])
def test_shared_results_match_standalone_metrics(
    real_df, syn_good_df, metadata, distance, with_holdout
):
    holdout = _holdout(real_df, with_holdout)
    report = _evaluate(real_df, syn_good_df, metadata, distance, holdout)
    by_name = {r.spec.name: r for r in report.results}
    for cls in _DISTANCE_METRICS:
        # Constructed without neighbors, each metric builds its own, as before #46.
        kwargs = {"holdout": holdout} if cls in (DCR, NNDR) else {}
        alone = cls(distance=distance, **kwargs).compute(real_df, syn_good_df, metadata)
        shared = by_name[cls.spec.name]
        assert shared.error is None, cls.spec.name
        assert shared.scalars == alone.scalars, cls.spec.name
        assert shared.per_column == alone.per_column, cls.spec.name
        assert shared.notes == alone.notes, cls.spec.name
        assert shared.skip_reason == alone.skip_reason, cls.spec.name
