from syneva.render.figures import BarComparison, HistogramOverlay


def test_histogram_overlay_renders_png():
    p = HistogramOverlay(real=[1, 2, 3, 4], synthetic=[2, 3, 4, 5], title="age")
    out = p.render_matplotlib()
    assert isinstance(out, bytes)
    assert out.startswith(b"\x89PNG")


def test_bar_comparison_renders_png():
    p = BarComparison(real={"M": 0.5, "F": 0.5}, synthetic={"M": 0.7, "F": 0.3}, title="sex")
    out = p.render_matplotlib()
    assert out.startswith(b"\x89PNG")
