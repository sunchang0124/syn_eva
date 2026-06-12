from __future__ import annotations

import base64
from pathlib import Path

import jinja2

from syneva.core.metric_info import (
    describe_c,
    describe_metric,
    display_name,
    humanize_scalar,
    verdict,
)
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
            scalars = r.scalars or {}
            # The headline score (normalized 0-1, 1=ideal) is shown prominently;
            # every other scalar is a raw supporting measurement under "Details".
            detail_rows = [(humanize_scalar(k), v) for k, v in scalars.items() if k != "score"]
            score = scalars.get("score")
            v_label, v_class = verdict(score) if score is not None else ("", "")
            entry = {
                "spec": r.spec,
                "display_name": display_name(r.spec.name),
                "score": score,
                "verdict_label": v_label,
                "verdict_class": v_class,
                "detail_rows": detail_rows,
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

    summary_cards = []
    for c, score in report.aggregated.items():
        v_label, v_class = verdict(score)
        summary_cards.append(
            {
                "c": c,
                "score": score,
                "verdict_label": v_label,
                "verdict_class": v_class,
                "info": describe_c(c),
            }
        )

    tpl = _ENV.get_template("scorecard.html.j2")
    return tpl.render(
        by_c=by_c_view,
        summary_cards=summary_cards,
        run_info=report.run_info,
    )
