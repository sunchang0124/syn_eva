from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class PlotPayload(Protocol):
    def render_matplotlib(self) -> bytes: ...
    def render_plotly(self) -> str: ...
