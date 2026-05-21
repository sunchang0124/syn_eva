from __future__ import annotations

import contextlib
import importlib.metadata
import time
from dataclasses import asdict, dataclass, field

_TRACKED = (
    "syneva",
    "pandas",
    "numpy",
    "scipy",
    "scikit-learn",
    "matplotlib",
    "jinja2",
    "typer",
    "pydantic",
)


@dataclass
class RunInfo:
    random_state: int
    started_at: float
    finished_at: float | None
    library_versions: dict[str, str]
    warnings: list[str] = field(default_factory=list)

    @classmethod
    def capture(cls, *, random_state: int) -> RunInfo:
        return cls(
            random_state=random_state,
            started_at=time.time(),
            finished_at=None,
            library_versions=_collect_versions(),
        )

    def finish(self) -> None:
        self.finished_at = time.time()

    def to_dict(self) -> dict:
        return asdict(self)


def _collect_versions() -> dict[str, str]:
    out: dict[str, str] = {}
    for name in _TRACKED:
        with contextlib.suppress(importlib.metadata.PackageNotFoundError):
            out[name] = importlib.metadata.version(name)
    return out
