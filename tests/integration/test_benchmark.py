import syneva


def test_benchmark_via_top_level(real_df, syn_good_df, syn_shifted_df, metadata):
    res = syneva.benchmark(real_df, {"good": syn_good_df, "shifted": syn_shifted_df}, metadata)
    ranked = res.ranking()
    assert ranked[0][1] == "good"
    assert isinstance(res, syneva.BenchmarkResult)
