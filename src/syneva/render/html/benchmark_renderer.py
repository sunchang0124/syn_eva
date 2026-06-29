from __future__ import annotations

from syneva.core.metric_info import describe_c, display_name, verdict
from syneva.render.html.renderer import _ENV, render_html


def render_leaderboard(
    result, *, normalization: str = "absolute", interactive: bool = False
) -> str:
    ranked = result.ranking(normalization=normalization)  # [(rank, name, key)]
    order = [name for _, name, _ in ranked]
    c_dims = result.c_dims()
    metrics = result.metrics()
    overall = result.overall
    c_scores = result.c_scores
    score_matrix = result.score_matrix

    def cell(score):
        if score is None:
            return {"score": None, "label": "", "cls": ""}
        v_label, v_class = verdict(score)
        return {"score": score, "label": v_label, "cls": v_class}

    leaderboard_rows = []
    for rank, name, _key in ranked:
        leaderboard_rows.append(
            {
                "rank": rank,
                "candidate": name,
                "overall": cell(overall.get(name)),
                "c_cells": {c: cell(c_scores.get(name, {}).get(c)) for c in c_dims},
            }
        )

    matrix_rows = []
    for name in order:
        row = score_matrix.get(name, {})
        matrix_rows.append({"candidate": name, "cells": {m: cell(row.get(m)) for m in metrics}})

    bars = []
    for name in order:
        bars.append(
            {
                "candidate": name,
                "c_bars": [{"c": c, **cell(c_scores.get(name, {}).get(c))} for c in c_dims],
            }
        )

    drilldowns = [
        {
            "candidate": name,
            "html": render_html(result.reports[name], interactive=interactive, fragment=True),
        }
        for name in order
    ]

    tpl = _ENV.get_template("leaderboard.html.j2")
    return tpl.render(
        normalization=normalization,
        c_dims=c_dims,
        c_info={c: describe_c(c) for c in c_dims},
        metrics=metrics,
        metric_names={m: display_name(m) for m in metrics},
        leaderboard_rows=leaderboard_rows,
        matrix_rows=matrix_rows,
        bars=bars,
        drilldowns=drilldowns,
        notes=result.notes,
    )
