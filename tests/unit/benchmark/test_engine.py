import json
import math
import sys

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


def test_empty_candidate_raises_before_any_evaluation(monkeypatch, real_df, syn_good_df, metadata):
    calls = []
    # syneva.benchmark is the function, so patch the engine module object itself
    engine = sys.modules[BenchmarkResult.__module__]
    monkeypatch.setattr(engine, "evaluate", lambda *a, **k: calls.append(a))
    with pytest.raises(SynevaError, match="candidate 'empty_cand' has no rows"):
        benchmark(real_df, {"ok": syn_good_df, "empty_cand": syn_good_df.iloc[:0]}, metadata)
    assert calls == []


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


def test_json_and_html_files_are_utf8(
    tmp_path, cp1252_locale, real_df, syn_good_df, syn_shifted_df, metadata
):
    from syneva.render.html.benchmark_renderer import render_leaderboard

    res = benchmark(real_df, {"候选 ≥50K": syn_good_df, "shifted": syn_shifted_df}, metadata)
    # a UTF-8 file with raw non-ASCII text, as another tool may write it
    path = tmp_path / "benchmark.json"
    path.write_bytes(json.dumps(res.to_dict(), ensure_ascii=False, default=str).encode("utf-8"))
    assert set(BenchmarkResult.from_json(path).reports) == {"候选 ≥50K", "shifted"}
    html = tmp_path / "lb.html"
    res.to_html(html)
    assert html.read_text(encoding="utf-8") == render_leaderboard(res)


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


# --- Candidates are compared only on metrics every candidate scored (#33) ---

_META = Metadata(columns={"x": ColumnMetadata(name="x", dtype=ColumnType.NUMERIC)})
_RUN_INFO = RunInfo(random_state=42, started_at=0.0, finished_at=1.0, library_versions={})


def _spec(name: str, c: str) -> MetricSpec:
    return MetricSpec(
        name=name,
        c=c,
        tier="core",
        data_types=frozenset(["static"]),
        requires_real=True,
        scope="table-level",
    )


def _report(**results: float | str | None) -> Report:
    """One result per `c__metric=value`: a float is a score, "skip" a skip, and
    None a failure."""
    out = []
    for key, value in results.items():
        c, _, name = key.partition("__")
        spec = _spec(name, c)
        if value is None:
            out.append(MetricResult(spec=spec, error=MetricError("simulated error")))
        elif value == "skip":
            out.append(MetricResult(spec=spec, skip_reason="simulated skip"))
        else:
            out.append(MetricResult(spec=spec, scalars={"score": value}))
    return Report(metadata=_META, results=out, run_info=_RUN_INFO)


_NORMALIZATIONS = ("absolute", "linear", "quantile", "normal")


@pytest.mark.parametrize("normalization", _NORMALIZATIONS)
def test_a_failed_metric_does_not_lift_a_candidate_above_one_that_scored_it(normalization):
    res = BenchmarkResult(
        reports={
            # Counting only what ran, A (0.8) would beat B ((0.9 + 0.2) / 2).
            "A": _report(compliance__dcr=0.8, compliance__mia_auc=None),
            "B": _report(compliance__dcr=0.9, compliance__mia_auc=0.2),
        }
    )
    assert res.overall == pytest.approx({"A": 0.8, "B": 0.9})
    assert [name for _, name, _ in res.ranking(normalization=normalization)] == ["B", "A"]


def test_a_dimension_failed_for_one_candidate_is_compared_for_none():
    res = BenchmarkResult(
        reports={
            "A": _report(congruence__ks=0.6, compliance__dcr=None),
            "B": _report(congruence__ks=0.7, compliance__dcr=0.1),
        }
    )
    assert res.c_dims() == ["congruence"]
    assert res.c_scores == {"A": {"congruence": 0.6}, "B": {"congruence": 0.7}}
    assert res.overall == pytest.approx({"A": 0.6, "B": 0.7})
    assert [name for _, name, _ in res.ranking()] == ["B", "A"]
    with pytest.raises(SynevaError, match="unknown ranking target"):
        res.ranking(by="compliance")


def test_a_metric_skipped_for_one_candidate_is_compared_for_none():
    res = BenchmarkResult(
        reports={
            "A": _report(congruence__ks=0.6, congruence__tvd="skip"),
            "B": _report(congruence__ks=0.7, congruence__tvd=0.1),
        }
    )
    assert res.compared_metrics() == ["ks"]
    assert [name for _, name, _ in res.ranking()] == ["B", "A"]


def test_a_candidate_with_nothing_scored_ranks_last_without_shrinking_the_comparison():
    res = BenchmarkResult(
        reports={
            "A": _report(congruence__ks=0.9, congruence__tvd=0.1),
            "B": _report(congruence__ks=0.5, congruence__tvd=0.6),
            "aaa_failed": _report(congruence__ks=None, congruence__tvd=None),
        }
    )
    assert res.compared_metrics() == ["ks", "tvd"]
    assert res.overall == pytest.approx({"A": 0.5, "B": 0.55, "aaa_failed": 0.0})
    assert res.c_scores["aaa_failed"] == {}
    assert [name for _, name, _ in res.ranking()] == ["B", "A", "aaa_failed"]


def test_ranking_by_one_metric_still_ranks_candidates_without_it_last():
    res = BenchmarkResult(
        reports={
            "A": _report(compliance__dcr=0.8, compliance__mia_auc=None),
            "B": _report(compliance__dcr=0.9, compliance__mia_auc=0.2),
        }
    )
    assert res.ranking(by="mia_auc") == [(1, "B", 0.2), (2, "A", float("-inf"))]


def test_notes_name_each_metric_left_out_of_the_comparison_once():
    res = BenchmarkResult(
        reports={
            "A": _report(congruence__ks=0.6, compliance__dcr=None, congruence__tvd=0.2),
            "B": _report(congruence__ks=0.7, compliance__dcr=0.1, congruence__tvd="skip"),
            "C": _report(congruence__ks=0.5, compliance__dcr="skip", congruence__tvd=0.3),
            "D": _report(congruence__ks=None, compliance__dcr=None, congruence__tvd=None),
        }
    )
    for _ in range(2):
        res.ranking()
        res.ranking(normalization="linear")
        _ = res.overall, res.c_scores
    assert res.notes == [
        "dcr not compared: failed for A, skipped for C",
        "tvd not compared: skipped for B",
        "D has no scored metrics and ranks last",
    ]


def test_a_metric_skipped_for_every_candidate_adds_no_note():
    res = BenchmarkResult(
        reports={
            "A": _report(congruence__ks=0.6, compliance__mia_auc="skip"),
            "B": _report(congruence__ks=0.7, compliance__mia_auc="skip"),
        }
    )
    res.ranking()
    assert res.notes == []


def test_comparison_notes_survive_a_json_round_trip(tmp_path):
    res = BenchmarkResult(
        reports={
            "A": _report(congruence__ks=0.6, compliance__dcr=None),
            "B": _report(congruence__ks=0.7, compliance__dcr=0.1),
        }
    )
    res.to_json(tmp_path / "b.json")
    loaded = BenchmarkResult.from_json(tmp_path / "b.json")
    assert loaded.notes == res.notes == ["dcr not compared: failed for A"]
    assert loaded.overall == res.overall
    assert loaded.c_scores == res.c_scores
    assert loaded.ranking() == res.ranking()


def test_with_one_candidate_scores_match_its_report():
    report = _report(congruence__ks=0.6, congruence__tvd=0.2, compliance__dcr=0.9)
    res = BenchmarkResult(reports={"only": report})
    assert res.c_scores == {"only": report.aggregated}
    assert res.overall["only"] == pytest.approx((0.4 + 0.9) / 2)


def test_a_metric_missing_from_a_report_is_noted_as_not_run():
    res = BenchmarkResult(
        reports={
            "A": _report(congruence__ks=0.6),
            "B": _report(congruence__ks=0.7, congruence__tvd=0.1),
        }
    )
    assert res.compared_metrics() == ["ks"]
    assert res.notes == ["tvd not compared: not run for A"]


def test_notes_passed_in_are_not_changed_in_place():
    mine = ["hand-written note"]
    res = BenchmarkResult(
        reports={
            "A": _report(congruence__ks=0.6, compliance__dcr=None),
            "B": _report(congruence__ks=0.7, compliance__dcr=0.1),
        },
        notes=mine,
    )
    assert mine == ["hand-written note"]
    assert res.notes == ["hand-written note", "dcr not compared: failed for A"]


def test_changing_the_returned_score_matrix_does_not_change_the_result():
    res = BenchmarkResult(reports={"A": _report(congruence__ks=0.6)})
    res.score_matrix["A"]["ks"] = 0.0
    assert res.score_matrix == {"A": {"ks": 0.6}}
    assert res.overall == {"A": 0.6}


def test_a_ranking_with_no_shared_metric_says_it_orders_by_name():
    res = BenchmarkResult(
        reports={
            "B": _report(congruence__ks=0.9, congruence__tvd="skip"),
            "A": _report(congruence__ks="skip", congruence__tvd=0.1),
        }
    )
    assert res.compared_metrics() == []
    assert [name for _, name, _ in res.ranking()] == ["A", "B"]
    assert "no metric was scored by every candidate, so candidates are ordered by name" in (
        res.notes
    )


def test_a_ranking_with_shared_metrics_adds_no_name_order_note():
    res = BenchmarkResult(
        reports={"A": _report(congruence__ks=0.6), "B": _report(congruence__ks=0.7)}
    )
    assert res.notes == []
