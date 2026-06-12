from syneva.coverage.extended.pca_scatter import PCAScatterMetric


def test_pca_scatter_attaches_plot(real_df, syn_good_df, metadata):
    r = PCAScatterMetric().compute(real_df, syn_good_df, metadata)
    assert r.scalars["score"] == 1.0  # visualization-only, score is neutral
    assert r.plot_payload is not None
    assert r.plot_payload.render_matplotlib().startswith(b"\x89PNG")
