import pandas as pd

from syneva.congruence.ks import KSStatistic
from syneva.congruence.tvd import TotalVariationDistance
from syneva.core.metadata import Metadata


def test_ks_attaches_histogram_overlay():
    df = pd.DataFrame({"x": list(range(100))})
    r = KSStatistic().compute(df, df, Metadata.infer(df))
    assert r.plot_payload is not None
    assert r.plot_payload.render_matplotlib().startswith(b"\x89PNG")


def test_tvd_attaches_bar_comparison():
    df = pd.DataFrame({"y": ["a", "b", "a", "b"] * 25})
    r = TotalVariationDistance().compute(df, df, Metadata.infer(df))
    assert r.plot_payload is not None
    assert r.plot_payload.render_matplotlib().startswith(b"\x89PNG")
