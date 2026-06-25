import math

import pytest

from syneva.benchmark.engine import BenchmarkResult, benchmark
from syneva.core.errors import MetricError, SynevaError
from syneva.core.metadata import ColumnMetadata, ColumnType, Metadata
from syneva.core.metric import MetricResult, MetricSpec
from syneva.core.report import Report
from syneva.core.run_info import RunInfo


def test_runs_each_candidate(real_df, syn_good_df, syn_shifted_df, metadata):
    res = benchmark(real_df, {"good": syn_good_df, "shifted": syn_shifted_df}, metadata)
    assert isinstance(res, BenchmarkResult)
    assert set(res.reports) == {"good", "shifted"}
    assert "good" in res.score_matrix and "shifted" in res.score_matrix


def test_overall_is_equal_weight_c_mean(real_df, syn_good_df, metadata):
    res = benchmark(real_df, {"good": syn_good_df}, metadata)
    cs = res.c_scores["good"]
    expected = sum(cs.values()) / len(cs)
    assert math.isclose(res.overall["good"], expected, rel_tol=1e-9)


def test_metrics_and_cdims_are_sorted_unions(real_df, syn_good_df, metadata):
    res = benchmark(real_df, {"good": syn_good_df}, metadata)
    assert res.metrics() == sorted(res.metrics())
    assert res.c_dims() == sorted(res.c_dims())


def test_empty_candidates_raises(real_df, metadata):
    with pytest.raises(SynevaError, match="at least one candidate"):
        benchmark(real_df, {}, metadata)


def test_schema_mismatch_candidate_raises(real_df, syn_good_df, metadata):
    bad = syn_good_df.drop(columns=[syn_good_df.columns[0]])
    with pytest.raises(SynevaError, match="bad_cand"):
        benchmark(real_df, {"ok": syn_good_df, "bad_cand": bad}, metadata)


def test_preset_forwarded_to_all(real_df, syn_good_df, syn_shifted_df, metadata):
    res = benchmark(
        real_df,
        {"good": syn_good_df, "shifted": syn_shifted_df},
        metadata,
        preset="privacy",
    )
    for name in ("good", "shifted"):
        assert set(res.c_scores[name]) == {"compliance"}


def test_ranking_better_candidate_wins_all_modes(real_df, syn_good_df, syn_shifted_df, metadata):
    res = benchmark(real_df, {"good": syn_good_df, "shifted": syn_shifted_df}, metadata)
    for mode in ("absolute", "linear", "normal", "quantile"):
        ranked = res.ranking(normalization=mode)
        assert ranked[0][1] == "good", f"mode={mode}"
        assert [r[0] for r in ranked] == [1, 2]


def test_ranking_by_single_c(real_df, syn_good_df, syn_shifted_df, metadata):
    res = benchmark(real_df, {"good": syn_good_df, "shifted": syn_shifted_df}, metadata)
    ranked = res.ranking(by="compliance")
    assert {r[1] for r in ranked} == {"good", "shifted"}


def test_ranking_unknown_by_raises(real_df, syn_good_df, metadata):
    res = benchmark(real_df, {"good": syn_good_df}, metadata)
    with pytest.raises(SynevaError, match="by"):
        res.ranking(by="not_a_thing")


def test_single_candidate_ranks_first_with_note(real_df, syn_good_df, metadata):
    res = benchmark(real_df, {"only": syn_good_df}, metadata)
    ranked = res.ranking(normalization="linear")
    assert ranked == [(1, "only", ranked[0][2])]


def test_roundtrip_dict_preserves_ranking(real_df, syn_good_df, syn_shifted_df, metadata):
    res = benchmark(real_df, {"good": syn_good_df, "shifted": syn_shifted_df}, metadata)
    res2 = BenchmarkResult.from_dict(res.to_dict())
    assert res2.ranking() == res.ranking()
    assert res2.overall == res.overall


def test_all_errored_candidate_ranks_last_under_relative_norm():
    """A candidate whose every metric errored must rank LAST, not tied with a
    genuine worst candidate that happens to normalize to key 0.0."""
    # Shared MetricSpec — same metric aligned across all three candidates
    spec = MetricSpec(
        name="test_metric",
        c="congruence",
        tier="core",
        data_types=frozenset(["static"]),
        requires_real=True,
        scope="table-level",
    )

    # Minimal metadata (one numeric column)
    meta = Metadata(columns={"x": ColumnMetadata(name="x", dtype=ColumnType.NUMERIC)})

    # Minimal RunInfo
    run_info = RunInfo(
        random_state=42,
        started_at=0.0,
        finished_at=1.0,
        library_versions={},
    )

    def make_report(score: float | None) -> Report:
        if score is None:
            # errored MetricResult — no usable score
            result = MetricResult(
                spec=spec,
                scalars=None,
                error=MetricError("simulated error"),
            )
        else:
            result = MetricResult(
                spec=spec,
                scalars={"score": score},
                error=None,
            )
        return Report(metadata=meta, results=[result], run_info=run_info)

    res = BenchmarkResult(
        reports={
            "best": make_report(0.9),
            "zworst": make_report(0.1),  # real worst — normalizes to key 0.0
            "aaa_err": make_report(None),  # all-errored; name sorts BEFORE "zworst"
        }
    )

    for mode in ("linear", "quantile", "normal", "absolute"):
        ranked = res.ranking(normalization=mode)
        names_in_order = [name for _, name, _ in ranked]
        assert names_in_order[0] == "best", (
            f"mode={mode}: expected 'best' first, got {names_in_order}"
        )
        assert names_in_order[-1] == "aaa_err", (
            f"mode={mode}: all-errored candidate 'aaa_err' must be last, got {names_in_order}"
        )
