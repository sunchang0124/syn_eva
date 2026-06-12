from __future__ import annotations

import io
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _to_png(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
    plt.close(fig)
    return buf.getvalue()


@dataclass
class HistogramOverlay:
    real: list[float]
    synthetic: list[float]
    title: str

    def render_matplotlib(self) -> bytes:
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.hist(self.real, bins=30, alpha=0.5, label="real")
        ax.hist(self.synthetic, bins=30, alpha=0.5, label="synthetic")
        ax.set_title(self.title)
        ax.legend(loc="best")
        return _to_png(fig)

    def render_plotly(self) -> str:  # pragma: no cover - import-gated
        import plotly.graph_objects as go

        fig = go.Figure()
        fig.add_histogram(x=self.real, name="real", opacity=0.5)
        fig.add_histogram(x=self.synthetic, name="synthetic", opacity=0.5)
        fig.update_layout(barmode="overlay", title=self.title)
        return fig.to_html(include_plotlyjs="cdn", full_html=False)


@dataclass
class BarComparison:
    real: dict[str, float]
    synthetic: dict[str, float]
    title: str

    def render_matplotlib(self) -> bytes:
        cats = sorted(set(self.real) | set(self.synthetic))
        xs = list(range(len(cats)))
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.bar(
            [x - 0.2 for x in xs],
            [self.real.get(c, 0.0) for c in cats],
            width=0.4,
            label="real",
        )
        ax.bar(
            [x + 0.2 for x in xs],
            [self.synthetic.get(c, 0.0) for c in cats],
            width=0.4,
            label="synthetic",
        )
        ax.set_xticks(xs, cats, rotation=30)
        ax.set_title(self.title)
        ax.legend(loc="best")
        return _to_png(fig)

    def render_plotly(self) -> str:  # pragma: no cover
        import plotly.graph_objects as go

        cats = sorted(set(self.real) | set(self.synthetic))
        fig = go.Figure()
        fig.add_bar(x=cats, y=[self.real.get(c, 0.0) for c in cats], name="real")
        fig.add_bar(x=cats, y=[self.synthetic.get(c, 0.0) for c in cats], name="synthetic")
        fig.update_layout(barmode="group", title=self.title)
        return fig.to_html(include_plotlyjs="cdn", full_html=False)
