import pytest

import syneva
import syneva.core.neighbors as neighbors
from syneva.compliance.dcr import DCR
from syneva.compliance.extended.epsilon_identifiability import EpsilonIdentifiability
from syneva.compliance.nndr import NNDR
from syneva.core.registry import registry
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
@pytest.mark.parametrize(("with_holdout", "expected"), [(False, 3), (True, 4)])
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
    # one, a single instance serves every gower metric and adds only holdout x real,
    # never holdout x holdout. (MIA is euclidean-only and builds its own.)
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
    report = _evaluate(real_df, syn_good_df, metadata, "euclidean", _holdout(real_df, with_holdout))
    # MIA encodes through the shared holdout instance too, so a holdout adds one.
    assert "mia_auc" in {r.spec.name for r in report.results}
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


# Scalars from evaluate() before Neighbors was shared (b907fd9), so a change inside
# Neighbors that affects shared and standalone runs alike still fails. The holdout is
# drawn from the real data, so its distances are 0.
_BEFORE_SHARING = {
    ("euclidean", False): {
        "alpha_precision_beta_recall": {
            "alpha_precision": 0.996,
            "beta_recall": 1.0,
            "score": 0.998,
        },
        "authenticity": {"authenticity": 0.972, "score": 0.972},
        "dcr": {
            "median_dcr": 0.18146004996286522,
            "p05_dcr": 0.05350233106469487,
            "p05_dcr_real": 0.166780976039137,
            "score": 0.3207939678452293,
        },
        "epsilon_identifiability": {"identifiability_risk": 0.9, "score": 0.09999999999999998},
        "nn_adversarial_accuracy": {
            "nn_adversarial_accuracy": 0.10200000000000001,
            "score": 0.20399999999999996,
        },
        "nndr": {"nndr_median": 0.44605493416751457, "score": 0.44605493416751457},
    },
    ("euclidean", True): {
        "alpha_precision_beta_recall": {
            "alpha_precision": 0.996,
            "beta_recall": 1.0,
            "score": 0.998,
        },
        "authenticity": {"authenticity": 0.972, "score": 0.972},
        "dcr": {
            "median_dcr": 0.18097115298458344,
            "median_dcr_holdout": 0.0,
            "p05_dcr": 0.053053376687700535,
            "p05_dcr_holdout": 0.0,
            "score": 1.0,
        },
        "epsilon_identifiability": {"identifiability_risk": 0.9, "score": 0.09999999999999998},
        "mia_auc": {"mia_auc": 0.46672, "score": 1.0},
        "nn_adversarial_accuracy": {
            "nn_adversarial_accuracy": 0.10200000000000001,
            "score": 0.20399999999999996,
        },
        "nndr": {"nndr_median": 0.4468594674757025, "nndr_median_holdout": 0.0, "score": 1.0},
    },
    ("gower", False): {
        "alpha_precision_beta_recall": {
            "alpha_precision": 0.992,
            "beta_recall": 0.998,
            "score": 0.995,
        },
        "authenticity": {"authenticity": 0.966, "score": 0.966},
        "dcr": {
            "median_dcr": 0.007761422983046403,
            "p05_dcr": 0.001488818891777039,
            "p05_dcr_real": 0.007075962576408571,
            "score": 0.21040513932914187,
        },
        "epsilon_identifiability": {"identifiability_risk": 0.888, "score": 0.11199999999999999},
        "nn_adversarial_accuracy": {"nn_adversarial_accuracy": 0.118, "score": 0.236},
        "nndr": {"nndr_median": 0.440662091042119, "score": 0.440662091042119},
    },
    ("gower", True): {
        "alpha_precision_beta_recall": {
            "alpha_precision": 0.992,
            "beta_recall": 0.998,
            "score": 0.995,
        },
        "authenticity": {"authenticity": 0.966, "score": 0.966},
        "dcr": {
            "median_dcr": 0.007761422983046403,
            "median_dcr_holdout": 0.0,
            "p05_dcr": 0.001488818891777039,
            "p05_dcr_holdout": 0.0,
            "score": 1.0,
        },
        "epsilon_identifiability": {"identifiability_risk": 0.888, "score": 0.11199999999999999},
        "mia_auc": {"mia_auc": 0.46672, "score": 1.0},
        "nn_adversarial_accuracy": {"nn_adversarial_accuracy": 0.118, "score": 0.236},
        "nndr": {"nndr_median": 0.440662091042119, "nndr_median_holdout": 0.0, "score": 1.0},
    },
}


@pytest.mark.slow
@pytest.mark.parametrize(("distance", "with_holdout"), list(_BEFORE_SHARING))
def test_shared_results_match_values_before_sharing(
    real_df, syn_good_df, metadata, distance, with_holdout
):
    report = _evaluate(real_df, syn_good_df, metadata, distance, _holdout(real_df, with_holdout))
    by_name = {r.spec.name: r for r in report.results}
    for name, expected in _BEFORE_SHARING[(distance, with_holdout)].items():
        assert by_name[name].error is None, name
        assert by_name[name].scalars == pytest.approx(expected, rel=1e-9, abs=0.0), name


def test_a_failed_shared_build_is_not_repeated(monkeypatch, real_df, syn_good_df, metadata):
    built = []

    def failing(self, *args, **kwargs):
        built.append(1)
        raise RuntimeError("cannot encode")

    monkeypatch.setattr(neighbors.Neighbors, "__init__", failing)
    report = _evaluate(real_df, syn_good_df, metadata, "euclidean", None)
    failed = [r for r in report.results if r.spec.name in _DISTANCE_METRIC_NAMES]
    assert len(failed) == len(_DISTANCE_METRICS)
    assert all("cannot encode" in str(r.error) for r in failed)
    # attribute_disclosure builds its own (quasi-identifier) Neighbors.
    assert len(built) == 2


def test_shared_neighbors_are_dropped_after_the_last_distance_metric(
    monkeypatch, real_df, syn_good_df, metadata
):
    import gc
    import weakref

    refs = []
    original_init = neighbors.Neighbors.__init__

    def spy(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        refs.append(weakref.ref(self))

    monkeypatch.setattr(neighbors.Neighbors, "__init__", spy)
    # novelty_rate runs after nn_adversarial_accuracy, the last distance metric.
    selected = registry.select(tiers=["core", "extended"], data_type="static", cs=None)
    novelty = next(c for c in selected if c.spec.name == "novelty_rate")
    alive_then = []
    original_compute = novelty.compute

    def compute(self, *args, **kwargs):
        gc.collect()
        alive_then.append(sum(r() is not None for r in refs))
        return original_compute(self, *args, **kwargs)

    monkeypatch.setattr(novelty, "compute", compute)
    _evaluate(real_df, syn_good_df, metadata, "euclidean", None)
    assert refs
    assert alive_then == [0]


_DISTANCE_METRIC_NAMES = {cls.spec.name for cls in _DISTANCE_METRICS}
