import pytest

from syneva.core.metadata import ColumnMetadata, ColumnType, Metadata
from syneva.core.metric import MetricResult, MetricSpec
from syneva.core.report import Report
from syneva.core.run_info import RunInfo


def _report():
    spec = MetricSpec(
        name="ks",
        c="congruence",
        tier="core",
        data_types=frozenset({"static"}),
        requires_real=True,
        scope="per-column",
    )
    meta = Metadata(columns={"x": ColumnMetadata(name="x", dtype=ColumnType.NUMERIC)})
    return Report(
        metadata=meta,
        results=[MetricResult(spec=spec, scalars={"score": 0.9})],
        run_info=RunInfo.capture(random_state=0),
    )


def test_pdf_renders(tmp_path):
    # weasyprint may be importable yet unusable when its native libraries
    # (e.g. libpango) are absent; importing it then raises OSError, not
    # ImportError, so importorskip alone is not enough.
    try:
        import weasyprint  # noqa: F401
    except (ImportError, OSError) as exc:
        pytest.skip(f"weasyprint unavailable: {exc}")
    out = tmp_path / "scorecard.pdf"
    _report().to_pdf(out)
    assert out.stat().st_size > 1000
    assert out.read_bytes()[:4] == b"%PDF"


def test_pdf_without_weasyprint_raises(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "weasyprint", None)
    from syneva.core.errors import SynevaError

    with pytest.raises(SynevaError, match="weasyprint"):
        _report().to_pdf("/tmp/x.pdf")
