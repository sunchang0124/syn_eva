from __future__ import annotations

from pathlib import Path

from syneva.core.errors import SynevaError
from syneva.core.report import Report
from syneva.render.html.renderer import render_html


def render_pdf(report: Report, path: str | Path) -> None:
    try:
        import weasyprint
    except (ImportError, OSError) as exc:
        raise SynevaError(
            "PDF rendering requires `pip install 'syneva[pdf]'` (weasyprint) "
            "and its native libraries (e.g. libpango)."
        ) from exc
    html = render_html(report, interactive=False)
    try:
        weasyprint.HTML(string=html).write_pdf(str(path))
    except OSError as exc:
        raise SynevaError(f"PDF rendering failed: {exc}") from exc
