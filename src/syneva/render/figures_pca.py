from __future__ import annotations

import io
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


@dataclass
class PCAScatter:
    real_xy: np.ndarray  # (N, 2)
    synthetic_xy: np.ndarray  # (M, 2)
    title: str = "PCA scatter"

    def render_matplotlib(self) -> bytes:
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(self.real_xy[:, 0], self.real_xy[:, 1], s=8, alpha=0.4, label="real")
        ax.scatter(
            self.synthetic_xy[:, 0],
            self.synthetic_xy[:, 1],
            s=8,
            alpha=0.4,
            label="synthetic",
        )
        ax.set_title(self.title)
        ax.legend(loc="best")
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
        plt.close(fig)
        return buf.getvalue()

    def render_plotly(self) -> str:  # pragma: no cover
        import plotly.graph_objects as go

        fig = go.Figure()
        fig.add_scatter(x=self.real_xy[:, 0], y=self.real_xy[:, 1], mode="markers", name="real")
        fig.add_scatter(
            x=self.synthetic_xy[:, 0],
            y=self.synthetic_xy[:, 1],
            mode="markers",
            name="synthetic",
        )
        fig.update_layout(title=self.title)
        return fig.to_html(include_plotlyjs="cdn", full_html=False)
