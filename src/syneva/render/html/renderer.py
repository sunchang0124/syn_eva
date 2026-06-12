from __future__ import annotations

import base64
from pathlib import Path

import jinja2

from syneva.core.metric_info import describe_c, describe_metric
from syneva.core.report import Report

_TEMPLATES = Path(__file__).parent / "templates"
_ENV = jinja2.Environment(
    loader=jinja2.FileSystemLoader(_TEMPLATES),
    autoescape=jinja2.select_autoescape(["html"]),
)


def render_html(report: Report, *, interactive: bool = False) -> str:
    by_c_view: dict[str, list] = {}
    for c, results in report.by_c.items():
        view = []
        for r in results:
            entry = {
                "spec": r.spec,
                "scalars": r.scalars,
                "notes": r.notes,
                "error": str(r.error) if r.error else None,
                "info": describe_metric(r.spec.name),
                "plot_b64": None,
                "plot_html": None,
            }
            if r.plot_payload is not None and r.error is None:
                if interactive:
                    entry["plot_html"] = r.plot_payload.render_plotly()
                else:
                    entry["plot_b64"] = base64.b64encode(r.plot_payload.render_matplotlib()).decode(
                        "ascii"
                    )
            view.append(entry)
        by_c_view[c] = view

    tpl = _ENV.get_template("scorecard.html.j2")
    return tpl.render(
        by_c=by_c_view,
        aggregated=report.aggregated,
        c_info={c: describe_c(c) for c in report.by_c},
        run_info=report.run_info,
    )
